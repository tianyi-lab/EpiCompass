"""EpiCompass on DiscoverPhysics (Wiemann et al., 2026): EpiDiscoveryAgent = the benchmark's DiscoveryAgent with the
epistemic state around the one call every turn makes (llm_client.complete); the agent's loop is untouched.

  - observations: every experiment of the round that just ran (from the agent's own conversation log);
  - hypotheses:   every law the agent sent to <run_mse_fit>, and its <final_law> (one per constant-stripped form);
  - feedback:     the epistemic feedback is appended to the environment message the agent is about to read;
  - selection:    at a hypothesis step K replies are sampled and one is selected by anticipated epistemic gain;
  - verification: the first <final_law> gets self-verification (the next submission is final).
Hypothesis check: the benchmark's own trajectory loss (the one behind <run_mse_fit>), per experiment, with the law's
declared parameters fitted on all experiments so far. Discriminating experiments call the law itself on inputs from the
world's own random-experiment sampler; the true simulator is never called.

DiscoverPhysics' ScienceAgent must be importable (see scripts/run_discoverphysics.py).
"""
from __future__ import annotations

import functools
import json
import signal
from typing import Any, Dict

import numpy as np

from scienceagent import agent as _agent_mod
from scienceagent.agent import DiscoveryAgent, _extract_tag

from .. import config
from ..baselines import make_session
from ..selection import choose, self_choose
from ._forms import family_signature

INF = float("inf")
_EXPLANATION_FOLLOWUP = "Your <final_law> submission is accepted"
FIT_CPU_S = 60.0
PROBE_CPU_S = 30.0


class _Timeout(BaseException):
    pass


class _cpu:
    def __init__(self, s):
        self.s = s

    def __enter__(self):
        def _raise(*_):
            raise _Timeout()
        self.old = signal.signal(signal.SIGVTALRM, _raise)
        signal.setitimer(signal.ITIMER_VIRTUAL, self.s)

    def __exit__(self, *e):
        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
        signal.signal(signal.SIGVTALRM, self.old)
        return False


class DPDomain:
    error_unit = "the trajectory mean-squared position error, the same loss as <run_mse_fit>"
    misfit_threshold = None     # positions carry noise of unknown scale: relative criteria only

    def __init__(self, agent):
        from scienceagent import mse_fitting as mf
        self.a, self.mf = agent, mf
        self.world = agent.trajectory_logger.world if agent.trajectory_logger else None
        self.loss_fn = mf._LOSS_FNS.get(self.world)
        self.two_particle = self.loss_fn is not None and self.loss_fn.__name__ == "_two_particle_loss"
        self._compiled: Dict[Any, Any] = {}
        self._params: Dict[Any, Any] = {}
        self._err: Dict[Any, float] = {}
        self._exps = (None, {})

    # data -------------------------------------------------------------------------------------------------------------
    def experiments(self):
        """{experiment_id: trajectory} from the run's own trajectory CSV; re-read only when the file changed."""
        import os
        from scienceagent.load_trajectories import load_trajectories
        tl = self.a.trajectory_logger
        try:
            stamp = (os.path.getmtime(tl.csv_path), os.path.getsize(tl.csv_path))
        except OSError:
            return {}
        if self._exps[0] != stamp:
            self._exps = (stamp, {e.experiment_id: e for e in load_trajectories(tl.csv_path, run_id=tl.run_id)})
        return self._exps[1]

    def key(self, artifact):
        return family_signature(artifact, "discovered_law")["family"]

    def _compile(self, artifact):
        k = hash(artifact)
        if k not in self._compiled:
            law = self.mf._compile_law(artifact)
            spec, fit_fn = [], self.mf._compile_fit_parameters(artifact)
            if fit_fn is not None:
                spec = self.mf._validate_fit_spec(fit_fn())
            self._compiled[k] = (law, spec)
        return self._compiled[k]

    def n_params(self, artifact):
        try:
            return len(self._compile(artifact)[1])
        except Exception:  # noqa: BLE001
            return 0

    def bound(self, artifact, obs, refit=True):
        """(law with its declared parameters fitted, parameter version). The benchmark optimiser, CPU-capped; with
        refit=False the last fit is reused (fitted on fewer observations) - the incremental belief update."""
        k = hash(artifact)
        law, spec = self._compile(artifact)
        cur = self._params.get(k)
        if cur is None or (refit and cur[1] != len(obs)):
            params = {n: init for n, init, _ in spec}
            if spec and obs:
                exps = self.experiments()
                tr = self.mf._experiments_to_training(self.world, [exps[o.inputs] for o in obs if o.inputs in exps])
                try:
                    with _cpu(FIT_CPU_S):
                        params = self.mf._fit_law_parameters(law, spec, tr, self.loss_fn)
                except (_Timeout, Exception):  # noqa: BLE001 -- keep the declared initial values
                    pass
            cur = (params, len(obs), (cur[2] + 1) if cur else 0)
            self._params[k] = cur
        return (functools.partial(law, **cur[0]) if cur[0] else law), cur[2]

    def errors(self, artifact, obs, refit=True):
        if self.loss_fn is None:
            return None
        b, ver = self.bound(artifact, obs, refit)
        exps = self.experiments()
        out = []
        try:
            with _cpu(FIT_CPU_S):
                for o in obs:
                    ck = (hash(artifact), ver, o.id)
                    if ck not in self._err:
                        e = exps.get(o.inputs)
                        try:
                            self._err[ck] = INF if e is None else float(
                                self.loss_fn(b, self.mf._experiments_to_training(self.world, [e])))
                        except Exception:  # noqa: BLE001
                            self._err[ck] = INF
                    out.append(self._err[ck])
        except _Timeout:
            return None                  # unknown (too slow to evaluate), not a failure
        return out

    def _pred(self, b, inp, t):
        p, _ = b(pos1=[0.0, 0.0], pos2=list(inp["pos2"]), p1=inp["p1"], p2=inp["p2"], velocity2=list(inp["velocity2"]),
                 duration=float(t))
        p = np.asarray(p, float)
        return p[-1] if p.ndim == 2 else p

    def _ref_inputs(self, obs, k=3):
        refs = []
        for o in obs[:k]:
            inp = o.outputs.get("_input") if isinstance(o.outputs, dict) else None
            if inp and all(x in inp for x in ("p1", "p2", "pos2", "velocity2")):
                ts = inp.get("measurement_times") or [inp.get("duration", 5.0)]
                refs.append((inp, float(max(ts))))
        return refs

    def discriminate(self, a, b, obs):
        if not self.two_particle:
            return []
        from scienceagent.random_experiments import make_random_generator
        try:
            gen = make_random_generator(self.world, seed=len(obs))
        except Exception:  # noqa: BLE001
            return []
        ba, bb = self.bound(a, obs)[0], self.bound(b, obs)[0]
        out = []
        try:
            with _cpu(PROBE_CPU_S):
                for _ in range(24):
                    inp = gen()
                    t = float(max(inp.get("measurement_times") or [inp.get("duration", 5.0)]))
                    pa, pb = self._pred(ba, inp, t), self._pred(bb, inp, t)
                    d = float(np.linalg.norm(pa - pb))
                    if np.isfinite(d):
                        shown = {k: inp[k] for k in ("p1", "p2", "pos2", "velocity2") if k in inp}
                        out.append({"input_text": json.dumps(shown, separators=(",", ":")), "what": f"position of particle 2 at t={t:g}",
                                    "pred_a": pa.tolist(), "pred_b": pb.tolist(), "divergence": d})
        except (_Timeout, Exception):  # noqa: BLE001
            pass
        out.sort(key=lambda x: -x["divergence"])
        return out[:2]

    # text -------------------------------------------------------------------------------------------------------------
    def describe_hyp(self, artifact):
        """What the law says (its core lines, card.code_core) and its free parameters as fitted on your experiments."""
        from ..card import code_core
        out = code_core(artifact)
        cur = self._params.get(hash(artifact))
        if cur and cur[0]:
            out.append("parameters fitted on your experiments: " + ", ".join(
                f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in list(cur[0].items())[:6]))
        return out

    def describe_obs(self, o):
        return f"experiment {o.label}"

    def compare(self, artifact, o):
        if not self.two_particle or not isinstance(o.outputs, dict) or "_input" not in o.outputs:
            return ""
        inp = o.outputs["_input"]
        try:
            ts = o.outputs.get("measurement_times") or []
            obs_p = o.outputs.get("pos2") or []
            if not ts or not obs_p:
                return ""
            b, _ = self.bound(artifact, self.a.epic.st.obs)
            with _cpu(PROBE_CPU_S):
                pr = self._pred(b, inp, ts[-1])
            return f"at t={ts[-1]:g} particle 2 was observed at ({obs_p[-1][0]:.3g}, {obs_p[-1][1]:.3g}); the law gives ({pr[0]:.3g}, {pr[1]:.3g})"
        except (_Timeout, Exception):  # noqa: BLE001
            return ""


class EpiDiscoveryAgent(DiscoveryAgent):
    """DiscoveryAgent + the epistemic state. method: epicompass | react | hsearch | popper; epic_model: the backbone
    name for the component's own calls (the agent's model without the provider prefix)."""

    def __init__(self, *args, epic_model=None, method=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method or config.METHOD
        self.epic_domain = DPDomain(self)
        self.epic_model = epic_model
        self.epic = make_session(self.epic_domain, epic_model, self.method)
        self.k = config.K if self.method in ("epicompass", "hsearch") else 1
        if self.method == "popper":
            def _p_examples():
                return [json.dumps({k: v for k, v in (o.outputs.get("_input") or {}).items()}, separators=(",", ":"))
                        for o in self.epic.st.obs if isinstance(o.outputs, dict)]

            def _p_run(turn, inp):
                if not isinstance(inp, dict):
                    return None, ""
                out = self.executor.run([inp])[0]
                if not isinstance(out, dict):
                    return None, ""
                k = 900 + self.epic.n_tests
                tl = self.trajectory_logger
                tl.log_experiment(round_num=k, source="popper", exp_input=inp, exp_output=out, exp_idx_in_round=0)
                shown = json.dumps({a: b for a, b in inp.items() if a != "measurement_times"}, separators=(",", ":"))
                o = self.epic.on_obs(turn, f"{tl.run_id}__r{k}__e0", dict(out, _input=inp),
                                     f"falsification test #{self.epic.n_tests + 1}: {shown[:160]}")
                return o, json.dumps(out)[:400]
            self.epic.io = (_p_examples, _p_run)
        self._seen = 0

    def _ingest(self):
        """Observations and <run_mse_fit> hypotheses from the rounds the loop has finished since the last call."""
        new = False
        run_id = self.trajectory_logger.run_id if self.trajectory_logger else "?"
        for entry in self.conversation_log[self._seen:]:
            r = entry.get("round")
            if entry.get("action") == "experiment" and entry.get("experiment_output"):
                for i, (inp, out) in enumerate(zip(entry.get("experiment_input") or [], entry["experiment_output"])):
                    if isinstance(out, dict):
                        out = dict(out, _input=inp)
                    shown = json.dumps({k: v for k, v in (inp or {}).items() if k != "measurement_times"}, separators=(",", ":"))
                    self.epic.on_obs(r, f"{run_id}__r{r}__e{i}", out, f"round {r} #{i + 1}: {shown[:160]}")
                    new = True
            if entry.get("mse_fit_input"):
                self.epic.on_hyp(r, "fit", entry["mse_fit_input"])
                new = True
        self._seen = len(self.conversation_log)
        return new

    def _complete(self, original, model, messages, system=None, max_tokens=None, **kw):
        if messages and messages[-1]["role"] == "user" and str(messages[-1]["content"]).startswith(_EXPLANATION_FOLLOWUP):
            return original(model=model, messages=messages, system=system, max_tokens=max_tokens, **kw)
        rnd = len(self.conversation_log) + 1
        try:
            notes = self.epic.round_notes(rnd) if self._ingest() else ""
        except Exception as e:  # noqa: BLE001 -- the component must never break the benchmark loop
            notes = ""
            self.epic.events.append({"turn": rnd, "event": "notes_error", "error": f"{type(e).__name__}: {str(e)[:200]}"})
        if notes:
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] = str(messages[-1]["content"]) + "\n" + notes
            else:
                messages.append({"role": "user", "content": notes})
        reply = self._pick(original, rnd, model=model, messages=messages, system=system, max_tokens=max_tokens, **kw)
        for _ in range(2):          # self-verification, then (once) the check of the revised submission
            fl = _extract_tag(reply or "", "final_law")
            if fl is None or rnd < self.min_rounds or self.epic.final_checked:
                break
            try:
                if not self.epic.reviewed:
                    txt = self.epic.review(rnd, fl.strip(), extra=f"You have {self.max_rounds - rnd} round(s) left.")
                else:
                    txt = self.epic.final_check(rnd, fl.strip())
            except Exception as e:  # noqa: BLE001
                txt = ""
                self.epic.events.append({"turn": rnd, "event": "review_error", "error": f"{type(e).__name__}: {str(e)[:200]}"})
            if not txt:
                break
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": txt.strip()})
            reply = self._pick(original, rnd, model=model, messages=messages, system=system, max_tokens=max_tokens, **kw)
        return reply

    @staticmethod
    def _hyp_of(reply):
        """The hypothesis a reply proposes: its <final_law>, else its <run_mse_fit> law (None if neither)."""
        for tag in ("final_law", "run_mse_fit"):
            x = _extract_tag(reply or "", tag)
            if x and "def " in x:
                return x.strip()
        return None

    def _pick(self, original, rnd, **call):
        """At a hypothesis step, K replies from the same context and the selected one becomes the agent's reply;
        otherwise a single call."""
        reply = original(**call)
        if self.k <= 1 or not self._hyp_of(reply):
            return reply
        hsearch = self.method == "hsearch"
        cands = [reply]
        for _ in range(self.k - 1):
            r = original(**call)
            if hsearch or self._hyp_of(r):            # Hypothesis Search: every sample is a candidate
                cands.append(r)
        if len(cands) == 1:
            return reply
        try:
            if hsearch:
                ctx = ([("system", call.get("system"))] if call.get("system") else []) + \
                    [(m.get("role"), m.get("content") if isinstance(m.get("content"), str) else str(m.get("content")))
                     for m in call.get("messages") or []]
                i, rec = self_choose(ctx, cands, self.epic_model, config.EFFORT)
            else:
                self._ingest()
                i, rec = choose(self.epic.st, self.epic_domain, [self._hyp_of(c) or c for c in cands], self.epic_model,
                                config.EFFORT)
        except Exception as e:  # noqa: BLE001
            i, rec = 0, {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        self.epic.events.append({"turn": rnd, "event": "select", "n_sampled": self.k, **rec})
        return cands[i]

    def run(self):
        original = _agent_mod.llm_client.complete
        _agent_mod.llm_client.complete = functools.partial(self._complete, original)
        try:
            law = super().run()
        finally:
            _agent_mod.llm_client.complete = original
        if law:
            try:
                self._ingest()
                self.epic.review(len(self.conversation_log), law)          # logs the final submission
            except Exception:  # noqa: BLE001
                pass
        return law

    def epic_record(self) -> Dict[str, Any]:
        return self.epic.record()
