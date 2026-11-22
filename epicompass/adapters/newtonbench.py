"""EpiCompass on NewtonBench (Zheng et al., 2026): the benchmark's own vanilla agent loop (its prompts, actions and
parsers, from NewtonBench's utils/vanilla_agent.py) with the epistemic state around it.

Hooks, and nothing else:
  - observations: every experiment (inputs, outcome, and the RNG seed it ran with: some modules draw nuisance factors
    from the global RNGs, so the replay uses the same draws);
  - hypotheses:   every `def discovered_law(...)` the agent writes in a reply, and its <final_law>. Constants are the
    agent's own here (the harness fits nothing), so a hypothesis is one exact law;
  - feedback:     the epistemic feedback is appended to the environment's reply after an experiment round;
  - selection:    at a hypothesis step K replies are sampled and one is selected by anticipated epistemic gain;
  - verification: the first <final_law> gets self-verification and the revised one a regression check; neither uses
    a turn (as in the vanilla loop, a reply with <final_law> otherwise ends the episode).
Hypothesis check: the law is put in place of the hidden law inside the module's OWN experiment function, the
experiment is re-run without noise with the same seed, and the outcome is compared with the observed one (relative
error). The ground truth is never read.

NewtonBench must be importable (its root on sys.path, e.g. via scripts/run_newtonbench.py --newtonbench_dir).
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import re
import signal
import sys
from typing import Any, Dict, List

from .. import config
from ..baselines import make_session
from ..llm import chat
from ..selection import choose, self_choose

from utils.vanilla_agent import BASE_PROMPT, _extract_final_law, parse_experiment_request  # noqa: E402  (NewtonBench)

INF = float("inf")
CALL_CPU_S = 10.0
EXACT = 1e-9          # relative errors below this are floating-point noise of an exact law (the data are noise-free)
GRID = (1e-3, 0.37, 1.7, 23.0, 410.0, 1e4, 1e8, 1e14)   # scale-free probes, used next to the agent's own points
N_POINTS = 6


class _Stop(BaseException):      # BaseException: a module's own `except Exception` must not swallow it
    pass


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


def _flat(series):
    try:
        import numpy as np
        return [float(x) for x in np.asarray(series, dtype=float).reshape(-1)]
    except Exception:  # noqa: BLE001
        return []


def _last(series):
    if isinstance(series, (list, tuple)):
        return _last(series[-1]) if series else float("nan")
    try:
        import numpy as np
        if isinstance(series, np.ndarray):
            return float(series.reshape(-1)[-1])
    except ImportError:
        pass
    return series


def _scalar(v):
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            v = v.reshape(-1)[0] if v.size else float("nan")
    except ImportError:
        pass
    if isinstance(v, (list, tuple)):
        v = v[0] if v else float("nan")
    return float(v)


def finite_outcome(res) -> bool:
    """The observed outcome carries evidence only if it is finite (the hidden law itself can overflow to nan)."""
    try:
        if isinstance(res, dict):
            vals = [x for series in res.values() for x in (series if isinstance(series, (list, tuple)) else
                                                               getattr(series, "tolist", lambda: [series])())]
            return bool(vals) and all(math.isfinite(float(_scalar(x))) for x in vals)
        return math.isfinite(_scalar(res))
    except Exception:  # noqa: BLE001
        return False


def _spread(xs, n):
    if len(xs) <= n:
        return list(xs)
    return [xs[round(i * (len(xs) - 1) / (n - 1))] for i in range(n)]


def law_args(signature: str) -> List[str]:
    m = re.search(r"def\s+discovered_law\s*\(([^)]*)\)", signature)
    return [a.split("=")[0].split(":")[0].strip() for a in (m.group(1).split(",") if m else []) if a.strip()]




# -- replaying a law inside the module's own experiment ------------------------------------------------------------------
def make_law(src: str):
    ns: Dict[str, Any] = {"math": math}
    try:
        import numpy as np
        ns["np"] = ns["numpy"] = np
    except ImportError:
        pass
    exec(compile(src, "<candidate>", "exec"), ns)  # same trust level as NewtonBench's own evaluate_law
    fn = ns.get("discovered_law")
    if not callable(fn):
        raise ValueError("the source defines no discovered_law(...)")
    return fn


def seed_rngs(seed: int) -> None:
    """Some modules draw nuisance factors from the global RNGs: the harness seeds them before the real experiment and
    again before the replay, so the two runs differ only in the law."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass


def predict(module: Any, law_fn, inputs: Dict[str, Any], difficulty: str, system: str, law_version,
            seed: int = None) -> Any:
    core = sys.modules[module.__name__ + ".core"]
    original = core.get_ground_truth_law
    core.get_ground_truth_law = lambda *_a, **_k: (law_fn, "candidate")
    if seed is not None:
        seed_rngs(seed)
    try:
        return module.run_experiment_for_module(**inputs, noise_level=0.0, difficulty=difficulty,
                                                system=system, law_version=law_version)
    finally:
        core.get_ground_truth_law = original


def _flatten(v) -> List[float]:
    if isinstance(v, (list, tuple)):
        return [x for item in v for x in _flatten(item)]
    try:
        return [float(v)]
    except Exception:  # noqa: BLE001
        return [float("nan")]


def outcome_error(pred: Any, actual: Any) -> float:
    """Scalar outcome: |p - a| / |a|. Time series: per observable, RMS(p - a) relative to how far the observed series
    moved from its first value; the largest observable counts."""
    if not isinstance(actual, dict):
        p, a = _flatten(pred)[0], _flatten(actual)[0]
        return abs(p - a) / (abs(a) + 1e-300) if math.isfinite(p) and math.isfinite(a) else float("nan")
    worst = 0.0
    for key, a_series in actual.items():
        a, p = _flatten(a_series), _flatten((pred or {}).get(key, []))
        if len(a) != len(p) or not a:
            return float("nan")
        pairs = [(x, y) for x, y in zip(p, a) if math.isfinite(y)]
        if any(not math.isfinite(x) for x, _ in pairs):
            return float("nan")
        if not pairs:
            continue
        rms_d = math.sqrt(sum((x - y) ** 2 for x, y in pairs) / len(pairs))
        moved = math.sqrt(sum((y - a[0]) ** 2 for _, y in pairs) / len(pairs))
        scale = moved if moved > 0 else (abs(a[0]) + 1e-300)
        worst = max(worst, rms_d / scale)
    return worst


def code_key(src: str) -> str:
    """The code without comments, layout and docstrings: two sources with the same key are the same program."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "raw:" + hashlib.sha1(src.encode()).hexdigest()
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            n.body = body[1:] or [ast.Pass()]
    return "ast:" + hashlib.sha1(ast.dump(tree, annotate_fields=False, include_attributes=False).encode()).hexdigest()


class NBDomain:
    error_unit = ("relative prediction error (your law put in place of the hidden one, each of your experiments re-run "
                  "with it and compared with the outcome you observed; 0 = exact)")

    def __init__(self, module, difficulty, system, law_version, noise_level):
        self.m, self.diff, self.sys, self.ver = module, difficulty, system, law_version
        self.args = law_args(module.FUNCTION_SIGNATURE)
        # noise 0: the experiments are exact, so a law that misses them on average by > 1% does not reproduce them
        self.misfit_threshold = 0.01 if not noise_level else None
        self._laws: Dict[int, Any] = {}
        self._err: Dict[tuple, float] = {}
        # the law's arguments at the agent's own experiments (what the module passes to the law for those inputs;
        # recorded with a stand-in law, so the hidden law is not called): probes at the scale the agent works at
        self.points: List[Dict[str, float]] = []
        self.probe = None                # frozen at the first hypothesis, so hypothesis keys stay comparable

    def record_point(self, inputs, seed):
        """The first call of the law, in the module's own experiment for these inputs, whose arguments are all finite
        and non-zero (a simulation may start from a degenerate state, e.g. a current ramping up from 0). The stand-in
        returns 0 and stops the experiment once it has a point (or after 200 calls)."""
        got, n = {}, [0]

        def stand_in(*a, **k):
            n[0] += 1
            try:
                pt = dict(zip(self.args, a), **k)
                pt = {x: _scalar(pt[x]) for x in self.args}
                if all(math.isfinite(v) and v != 0 for v in pt.values()):
                    got.update(pt)
            except Exception:  # noqa: BLE001
                pass
            if got or n[0] >= 200:
                raise _Stop()
            return 0.0
        try:
            self._predict(stand_in, inputs, seed)
        except (_Stop, _Timeout, Exception):  # noqa: BLE001
            pass
        if got and got not in self.points:
            self.points.append(dict(got))
            return dict(got)
        return None

    def _grid(self):
        return [{a: b * (1 + 0.13 * i) for i, a in enumerate(self.args)} for b in GRID]

    def _law(self, artifact):
        k = hash(artifact)
        if k not in self._laws:
            try:
                self._laws[k] = make_law(artifact)
            except Exception:  # noqa: BLE001
                self._laws[k] = None
        return self._laws[k]

    def key(self, artifact):
        """Same function = same hypothesis: two programs that agree to 6 significant digits at the agent's own
        experiment points and on a scale-free grid (e.g. `m1*m2/r**2` and `m2*m1*r**-2`) are one law. Program
        identity (code_key) when there are no experiment points yet, the law cannot be evaluated, or it is nearly
        constant on the probes (then agreement there says nothing)."""
        fn = self._law(artifact)
        if self.probe is None and self.points:
            self.probe = _spread(self.points, N_POINTS)
        if fn is None or not self.args or self.probe is None:
            return code_key(artifact)
        vals = []
        for x in self.probe + self._grid():
            try:
                with _cpu(CALL_CPU_S):
                    v = float(fn(**x))
                vals.append(f"{v:.6g}" if math.isfinite(v) else "nf")
            except (_Timeout, Exception):  # noqa: BLE001
                vals.append("err")
        if len({v for v in vals if v not in ("nf", "err")}) < 3:
            return code_key(artifact)
        return "fn:" + "|".join(vals)

    def n_params(self, artifact):
        return 0

    def _predict(self, fn, inputs, seed):
        with _cpu(CALL_CPU_S):
            return predict(self.m, fn, inputs, self.diff, self.sys, self.ver, seed=seed)

    def errors(self, artifact, obs, refit=True):
        fn = self._law(artifact)
        out = []
        for o in obs:
            k = (hash(artifact), o.id)
            if k not in self._err:
                if fn is None:
                    self._err[k] = INF
                else:
                    try:
                        e = outcome_error(self._predict(fn, o.inputs, o.outputs["seed"]), o.outputs["result"])
                        self._err[k] = (0.0 if e < EXACT else e) if math.isfinite(e) else INF
                    except (_Timeout, Exception):  # noqa: BLE001
                        self._err[k] = INF
            out.append(self._err[k])
        return out

    def discriminate(self, a, b, obs):
        fa, fb = self._law(a), self._law(b)
        if fa is None or fb is None or not obs:
            return []
        keys = [k for k, v in obs[0].inputs.items() if isinstance(v, (int, float))]
        if not keys:
            return []
        lo = {k: min(float(o.inputs[k]) for o in obs if k in o.inputs) for k in keys}
        hi = {k: max(float(o.inputs[k]) for o in obs if k in o.inputs) for k in keys}
        ints = {k: all(isinstance(o.inputs.get(k), int) for o in obs) for k in keys}
        rng = random.Random(len(obs))
        done = {json.dumps(o.inputs, sort_keys=True) for o in obs}
        out = []
        for _ in range(16):
            x = dict(obs[0].inputs)
            for k in keys:
                l, h = lo[k] / 2 if lo[k] > 0 else lo[k] * 2, hi[k] * 2 if hi[k] > 0 else hi[k] / 2
                x[k] = (math.exp(rng.uniform(math.log(l), math.log(h))) if l > 0 and h > l else rng.uniform(min(l, h), max(l, h)))
                x[k] = int(round(x[k])) if ints[k] else float(f"{x[k]:.4g}")
            if json.dumps(x, sort_keys=True) in done:
                continue
            try:
                pa, pb = self._predict(fa, x, 12345), self._predict(fb, x, 12345)
                d = outcome_error(pa, pb)
            except (_Timeout, Exception):  # noqa: BLE001
                continue
            if not math.isfinite(d):
                continue
            what, va, vb = "outcome", pa, pb
            if isinstance(pa, dict) and isinstance(pb, dict):
                try:
                    kk = max((k for k in pb if k in pa), key=lambda k: outcome_error({k: pa[k]}, {k: pb[k]}))
                    what, va, vb = f"{kk} (last recorded value)", _last(pa[kk]), _last(pb[kk])
                except Exception:  # noqa: BLE001
                    continue
            try:
                va, vb = float(va), float(vb)
            except Exception:  # noqa: BLE001
                continue
            out.append({"input_text": json.dumps(x, separators=(",", ":")), "what": what, "pred_a": va, "pred_b": vb,
                        "divergence": d})
        out.sort(key=lambda z: -z["divergence"])
        return [z for z in out if z["divergence"] > 0.01][:2]

    def describe_obs(self, o):
        return f"experiment {o.label}"

    def compare(self, artifact, o):
        fn = self._law(artifact)
        if fn is None:
            return ""
        try:
            p, a = self._predict(fn, o.inputs, o.outputs["seed"]), o.outputs["result"]
            if not isinstance(a, dict):
                return f"observed {float(a):.6g}, the law gives {float(p):.6g}"
            k = max((k for k in a if k in (p or {})), key=lambda k: outcome_error({k: p[k]}, {k: a[k]}))
            # the recorded point where law and observation differ most (the last one may agree, e.g. both 0)
            fa, fp = _flat(a[k]), _flat(p[k])
            if len(fa) == len(fp) and fa:
                i = max(range(len(fa)), key=lambda j: abs(fp[j] - fa[j]))
                if len(fa) > 1:
                    # the error is relative to how far the observed series moved from its start, so show that
                    # motion: the change since the first recorded point, observed vs the law
                    return (f"{k} at recorded point {i + 1} of {len(fa)}: change since its start value "
                            f"{fa[0]:.6g} is {fa[i] - fa[0]:+.4g} observed, {fp[i] - fp[0]:+.4g} under the law")
                return f"{k}: observed {fa[i]:.6g}, the law gives {fp[i]:.6g}"
            return f"last recorded {k}: observed {float(_last(a[k])):.6g}, the law gives {float(_last(p[k])):.6g}"
        except (_Timeout, Exception):  # noqa: BLE001
            return "the law gives no finite value"

_LAW_IN_TEXT = re.compile(r"(def discovered_law\s*\(.*?)(?=\n\S|\n```|\Z)", re.S)


def _parses(src):
    """A law written in the agent's text counts as a hypothesis only if it is Python (not prose that mentions one)."""
    import ast
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def run_episode(module: Any, model: str, noise_level: float = 0.0, difficulty: str = "easy",
                system: str = "vanilla_equation", law_version: str = None, max_turns: int = 10,
                method: str = None, temperature: float = 0.4) -> Dict[str, Any]:
    """One NewtonBench episode with `method` (epicompass | react | hsearch | popper); returns the exploration result in
    NewtonBench's format (submitted_law, chat_history, ...) plus the epistemic-state record."""
    method = method or config.METHOD
    dom = NBDomain(module, difficulty, system, law_version, noise_level)
    es = make_session(dom, model, method)
    k = config.K if method in ("epicompass", "hsearch") else 1
    if method == "popper":
        def _p_examples():
            return [json.dumps(o.inputs, separators=(",", ":")) for o in es.st.obs]

        def _p_run(turn, inp):
            if not isinstance(inp, dict):
                return None, ""
            seed = int.from_bytes(os.urandom(4), "little")
            seed_rngs(seed)
            res = module.run_experiment_for_module(**inp, noise_level=noise_level, difficulty=difficulty,
                                                   system=system, law_version=law_version)
            if not finite_outcome(res):
                return None, ""
            dom.record_point(dict(inp), seed)
            o = es.on_obs(turn, dict(inp), {"result": res, "seed": seed}, json.dumps(inp, separators=(",", ":"))[:160])
            return o, f"output {json.dumps(res)[:400] if not isinstance(res, float) else f'{res:.6g}'}"
        es.io = (_p_examples, _p_run)

    messages = [{"role": "system", "content": BASE_PROMPT.format(max_turns=max_turns)},
                {"role": "user", "content": module.get_task_prompt(system, noise_level=noise_level)}]
    total_tokens = 0
    num_experiments = 0

    def result(status, law, rounds):
        return {"status": status, "submitted_law": law, "rounds": rounds, "total_tokens": total_tokens,
                "num_experiments": num_experiments, "chat_history": messages, **es.record()}

    def sample():
        """One reply from the current context, not yet committed to the conversation."""
        txt, usage = chat(model, messages, config.EFFORT, temperature=temperature)
        return txt or "", int((usage or {}).get("total_tokens") or 0)

    def hyp_of(text):
        """The hypothesis a reply proposes: its submitted law, else the last law written in its text (None if none)."""
        ok, law = _extract_final_law(text, module.FUNCTION_SIGNATURE)
        if ok:
            return law
        laws = [x for x in _LAW_IN_TEXT.findall(text) if "return" in x and _parses(x)]
        return laws[-1] if laws else None

    def call(turn):
        nonlocal total_tokens
        text, tok = sample()
        total_tokens += tok
        if k > 1 and hyp_of(text):
            # a hypothesis step: K samples from the same context, one selected
            cands = [(text, hyp_of(text))]
            for _ in range(k - 1):
                t, tok = sample()
                total_tokens += tok
                if method == "hsearch" or hyp_of(t):      # Hypothesis Search: every sample is a candidate
                    cands.append((t, hyp_of(t) or t))
            if len(cands) > 1:
                try:
                    if method == "hsearch":
                        ctx = [(m.get("role"), m.get("content") or "") for m in messages]
                        i, rec = self_choose(ctx, [z[0] for z in cands], model, config.EFFORT)
                    else:
                        i, rec = choose(es.st, dom, [z[1] for z in cands], model, config.EFFORT)
                except Exception as e:  # noqa: BLE001 -- the component must never break the benchmark loop
                    i, rec = 0, {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                es.events.append({"turn": turn, "event": "select", "n_sampled": k, **rec})
                text = cands[i][0]
        messages.append({"role": "assistant", "content": text})
        final_part = text[text.rfind("<final_law>"):] if "<final_law>" in text else ""
        for src in _LAW_IN_TEXT.findall(text.replace(final_part, "")):
            if "return" in src and _parses(src):
                es.on_hyp(turn, "text", src)
        return text

    def gate(turn, text, turns_left):
        """Self-verification / regression check of a <final_law>: the text to show, or "" to accept the submission."""
        ok, law = _extract_final_law(text, module.FUNCTION_SIGNATURE)
        if not ok or es.final_checked:
            return ""
        try:
            if not es.reviewed:
                return es.review(turn, law, extra=f"You have {turns_left} experiment turn(s) left.")
            return es.final_check(turn, law)
        except Exception as e:  # noqa: BLE001
            es.events.append({"turn": turn, "event": "review_error", "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return ""

    turn = 0
    while turn < max_turns:
        text = call(turn + 1)
        is_submitted, submitted_law = _extract_final_law(text, module.FUNCTION_SIGNATURE)
        if is_submitted:
            note = gate(turn + 1, text, max_turns - turn - 1)
            if not note:
                if es.reviewed:
                    es.review(turn + 1, submitted_law)          # logs the final submission
                return result("completed", submitted_law, turn + 1)
            messages.append({"role": "user", "content": note.strip()})
            continue                                            # self-verification does not use a turn

        experiments = parse_experiment_request(text)
        if experiments:
            num_experiments += len(experiments)
            results = []
            for exp in experiments:
                seed = int.from_bytes(os.urandom(4), "little")
                seed_rngs(seed)
                res = module.run_experiment_for_module(**exp, noise_level=noise_level, difficulty=difficulty,
                                                       system=system, law_version=law_version)
                if finite_outcome(res):
                    dom.record_point(dict(exp), seed)
                    es.on_obs(turn + 1, dict(exp), {"result": res, "seed": seed},
                              json.dumps(exp, separators=(",", ":"))[:160])
                else:
                    es.events.append({"turn": turn + 1, "event": "obs_skipped", "why": "non-finite outcome"})
                results.append("{:.15e}".format(res) if system == "vanilla_equation" else res)
            content = f"<experiment_output>\n{json.dumps(results)}\n</experiment_output>"
        else:
            content = ("Invalid response. Please use <run_experiment> tag with the correct JSON format or <final_law> "
                       "tag to submit the law.")
        try:
            notes = es.round_notes(turn + 1)
        except Exception as e:  # noqa: BLE001
            notes = ""
            es.events.append({"turn": turn + 1, "event": "notes_error", "error": f"{type(e).__name__}: {str(e)[:200]}"})
        messages.append({"role": "user", "content": content + ("\n" + notes if notes else "")})
        turn += 1

    final_prompt = "You have used all your experiment turns. Please submit your final law now using the <final_law> tag."
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] += "\n\n" + final_prompt
    else:
        messages.append({"role": "user", "content": final_prompt})
    text = call(max_turns + 1)
    for _ in range(2):
        note = gate(max_turns + 1, text, 0)
        if not note:
            break
        messages.append({"role": "user", "content": note.strip() + "\nThis is still your last action: reply with one <final_law>."})
        text = call(max_turns + 1)
    _, submitted_law = _extract_final_law(text, module.FUNCTION_SIGNATURE)
    if es.reviewed:
        es.review(max_turns + 1, submitted_law)
    return result("max_turns_reached", submitted_law, max_turns)
