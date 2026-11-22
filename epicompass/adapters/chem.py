"""EpiCompass on ActiveSciBench-Chem (LLM-AutoSciLab, Kabra et al., 2026): enzyme-kinetics rate-law discovery with the
benchmark's own oracle (ChemBenchOracle: run / evaluate_law). The benchmark's LLM-only agent is not released, so the
agent loop is a NewtonBench-style vanilla agent (<run_experiment> JSON list -> <experiment_output>, <final_law> ends the
task), the same for every method; the component sits around its LLM call as on NewtonBench:
  observations: every experiment (inputs -> r0, the measured initial rate);
  hypotheses:   every discovered_law written in a reply (the submitted <final_law> included);
  feedback:     appended to the experiment output; self-verification of the first <final_law>;
  selection:    at a step that writes a law, K replies are sampled and one is selected by anticipated epistemic gain.
Hypothesis check: relative error of r0 per experiment. The oracle's data carry deterministic secondary T/pH effects of
at most ~5% (the evaluation compares with the clean primary law), so the conflict threshold is 6%.
Task setting as the released runner: noise 0, grammar "universal". LLM-AutoSciLab must be importable
(see scripts/run_chem.py).
"""
from __future__ import annotations

import json
import math
import re
import signal

import numpy as np

from .. import config
from ..baselines import make_session
from ..llm import chat
from ..selection import choose, self_choose

INF = float("inf")
CALL_S = 2
ARGS = ["C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH"]
MAX_ROUNDS, BUDGET, PER_ROUND = 12, 60, 10
_LAW_IN_TEXT = re.compile(r"(def discovered_law\s*\(.*?)(?=\n\S|\n```|\Z)", re.S)
HINT = ("\nNote: the measured r0 includes small background effects of temperature and pH (at most about +-5% overall) "
        "that are not part of the rate law to discover; your law is scored against the rate without them.")

SYSTEM = """You are an AI research assistant tasked with discovering the rate law of an enzyme-catalyzed reaction.
Your goal is to propose experiments, analyze the data they return, and ultimately deduce the underlying rate law.
You can perform experiments to gather data but you must follow the protocol strictly.

**Workflow:**
1. Analyze the mission description provided.
2. Design a set of experiments to test your hypotheses.
3. Use the <run_experiment> tag to submit your experimental inputs: a JSON list of up to {per_round} experiments, each a
   dict with all seven inputs, e.g. <run_experiment>[{{"C_A": 1.0, "C_I": 0.0, "C_B": 1.0, "C_P": 0.0, "Enz": 1.0,
   "T": 310.0, "pH": 7.0}}]</run_experiment>
4. The system returns the measured outputs in an <experiment_output> tag.
5. You can run up to {rounds} rounds of experiments and at most {budget} experiments in total. Use them wisely.
6. Only one action is allowed per round: either <run_experiment> or <final_law>.
7. You should verify your hypotheses by checking whether the output of your hypothesis matches the measured data.
8. When confident, submit your final rate law as a Python function in the <final_law> tag, with exactly this signature:
   {signature}
       ...
       return r0
   The law should give the initial rate r0 [mM/min]. This ends the mission."""


class _TO(BaseException):
    pass


def _alarm(*_):
    raise _TO()


def law_of(text):
    """The law a reply writes: its <final_law>, else the last parseable discovered_law in the text (None if none)."""
    m = re.findall(r"<final_law>(.*?)</final_law>", text or "", re.S)
    if m:
        src = re.sub(r"^```(python)?|```$", "", m[-1].strip()).strip()
        return src if "def discovered_law" in src else None
    xs = []
    for x in _LAW_IN_TEXT.findall(text or ""):
        try:
            compile(x, "<law>", "exec")
            if "return" in x:
                xs.append(x)
        except SyntaxError:
            pass
    return xs[-1] if xs else None


def experiments_of(text):
    a = (text or "").rfind("<run_experiment>")
    if a < 0:
        return []
    b = text.find("</run_experiment>", a)
    if b < 0:
        return []
    try:
        x = json.loads(text[a + len("<run_experiment>"):b].strip())
    except Exception:  # noqa: BLE001
        return []
    x = x if isinstance(x, list) else [x]
    return [e for e in x if isinstance(e, dict)]


class ChemDomain:
    error_unit = "relative error |law r0 - measured r0| / |measured r0| on each of your experiments (0 = exact)"
    misfit_threshold = 0.06

    def __init__(self):
        self._fn = {}
        self.points = []

    def _compile(self, artifact):
        if artifact not in self._fn:
            ns = {"np": np, "numpy": np, "math": math}
            try:
                exec(compile(artifact, "<law>", "exec"), ns)
                self._fn[artifact] = ns.get("discovered_law")
            except Exception:  # noqa: BLE001
                self._fn[artifact] = None
        return self._fn[artifact]

    def _pred(self, artifact, x):
        fn = self._compile(artifact)
        if fn is None:
            return None
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(CALL_S)
        try:
            with np.errstate(all="ignore"):
                v = float(fn(*[float(x[a]) for a in ARGS]))
            return v if math.isfinite(v) else None
        except (_TO, Exception):  # noqa: BLE001
            return None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def key(self, artifact):
        probes = self.points[:8] + [{"C_A": 0.5 * 3 ** i, "C_I": 2.0 * i, "C_B": 1.0 + i, "C_P": 1.0 * i, "Enz": 1.0,
                                     "T": 300.0 + 10 * i, "pH": 6.0 + 0.5 * i} for i in range(4)]
        vals = [self._pred(artifact, x) for x in probes]
        vals = ["x" if v is None else f"{v:.6g}" for v in vals]
        return "src:" + re.sub(r"\s+", "", artifact) if len(set(vals)) < 2 else "fn:" + "|".join(vals)

    def n_params(self, artifact):
        return 0

    def errors(self, artifact, obs, refit=True):
        out = []
        for o in obs:
            p, y = self._pred(artifact, o.inputs), float(o.outputs)
            if p is None:
                out.append(INF)
                continue
            e = abs(p - y) / max(abs(y), 1e-300)
            out.append(0.0 if e < 1e-9 else e)
        return out

    def discriminate(self, a, b, obs):
        return []

    def describe_obs(self, o):
        return f"experiment {o.label}"

    def compare(self, artifact, o):
        p = self._pred(artifact, o.inputs)
        return "the law gives no finite value" if p is None else f"measured r0 {float(o.outputs):.6g}, the law gives {p:.6g}"


def _fmt(res):
    out = res.metadata.get("outputs", {})
    return {"r0": float(f"{res.measurement:.6g}"), **{k: float(f"{v:.6g}") for k, v in out.items() if k != "r0"}}


def run_task(domain, difficulty, law_version, model, method=None, hint=False, max_tokens=None):
    """One task; returns the result record (final law, evaluation, conversation, epistemic-state record)."""
    from autoscilab.oracle.chembench import ChemBenchOracle
    method = method or config.METHOD
    oracle = ChemBenchOracle(domain, difficulty=difficulty, noise_level=0.0, law_version=law_version,
                             grammar_mode="universal")
    dom = ChemDomain()
    es = make_session(dom, model, method)
    k = config.K if method in ("epicompass", "hsearch") else 1
    if method == "popper":
        def _p_examples():
            return [json.dumps(o.inputs, separators=(",", ":")) for o in es.st.obs]

        def _p_run(turn, inp):
            if not isinstance(inp, dict) or not all(a in inp for a in ARGS):
                return None, ""
            res = oracle.run({a: float(inp[a]) for a in ARGS})
            dom.points.append({a: float(inp[a]) for a in ARGS})
            o = es.on_obs(turn, {a: float(inp[a]) for a in ARGS}, float(res.measurement),
                          json.dumps(inp, separators=(",", ":"))[:160])
            return o, f"r0 {res.measurement:.6g}"
        es.io = (_p_examples, _p_run)
    mission = (f"Mission: discover the rate law r0 = f(C_A, C_I, C_B, C_P, Enz, T, pH) of an enzyme-catalyzed "
               f"reaction.\n{oracle.param_description}\nOutputs of an experiment: r0 (the measured initial rate) and the "
               f"concentrations after the fixed reaction time and the conversion." + (HINT if hint else ""))
    messages = [{"role": "system", "content": SYSTEM.format(per_round=PER_ROUND, rounds=MAX_ROUNDS, budget=BUDGET,
                                                             signature=oracle.function_signature)},
                {"role": "user", "content": mission}]
    used, tokens, turn, final = 0, 0, 0, None
    extra = {"max_tokens": max_tokens} if max_tokens else {}

    def sample():
        txt, u = chat(model, messages, config.EFFORT, timeout=1800, **extra)
        return txt or "", int((u or {}).get("total_tokens") or 0)

    def call(t):
        nonlocal tokens
        txt, n = sample()
        tokens += n
        if k > 1 and law_of(txt):
            cands = [txt]
            for _ in range(k - 1):
                t2, n2 = sample()
                tokens += n2
                if method == "hsearch" or law_of(t2):
                    cands.append(t2)
            if len(cands) > 1:
                try:
                    if method == "hsearch":
                        i, rec = self_choose([(m["role"], m["content"]) for m in messages], cands, model, config.EFFORT)
                    else:
                        i, rec = choose(es.st, dom, [law_of(c) or c for c in cands], model, config.EFFORT)
                except Exception as e:  # noqa: BLE001 -- the component must never break the benchmark loop
                    i, rec = 0, {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                es.events.append({"turn": t, "event": "select", "n_sampled": k, **rec})
                txt = cands[i]
        messages.append({"role": "assistant", "content": txt})
        law = law_of(txt)
        if law and "<final_law>" not in txt:
            es.on_hyp(t, "text", law)
        return txt

    def gate(t, law, left):
        if es.final_checked:
            return ""
        try:
            if not es.reviewed:
                return es.review(t, law, extra=f"You have {left} experiment round(s) left.")
            return es.final_check(t, law)
        except Exception as e:  # noqa: BLE001
            es.events.append({"turn": t, "event": "review_error", "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return ""

    while turn < MAX_ROUNDS:
        txt = call(turn + 1)
        if "<final_law>" in txt and law_of(txt):
            note = gate(turn + 1, law_of(txt), MAX_ROUNDS - turn - 1)
            if not note:
                final = law_of(txt)
                if es.reviewed:
                    es.review(turn + 1, final)
                break
            messages.append({"role": "user", "content": note.strip()})
            continue
        exps = experiments_of(txt)[:PER_ROUND]
        exps = exps[:max(0, BUDGET - used)]
        if exps:
            rows = []
            for e in exps:
                try:
                    x = {a: float(e[a]) for a in ARGS}
                    res = oracle.run(x)
                    rows.append(_fmt(res))
                    used += 1
                    dom.points.append(x)
                    es.on_obs(turn + 1, x, float(res.measurement), json.dumps(x, separators=(",", ":"))[:160])
                except Exception as ex:  # noqa: BLE001
                    rows.append({"error": f"{type(ex).__name__}: {str(ex)[:120]}"})
            content = (f"<experiment_output>\n{json.dumps(rows)}\n</experiment_output>\n"
                       f"Experiments used: {used}/{BUDGET}.")
        elif used >= BUDGET:
            content = "Your experiment budget is used up. Submit your final law with the <final_law> tag."
        else:
            content = ("Invalid response. Please use the <run_experiment> tag with a JSON list, or the <final_law> tag "
                       "to submit the law.")
        try:
            notes = es.round_notes(turn + 1)
        except Exception as e:  # noqa: BLE001
            notes = ""
            es.events.append({"turn": turn + 1, "event": "notes_error", "error": f"{type(e).__name__}: {str(e)[:200]}"})
        messages.append({"role": "user", "content": content + ("\n" + notes if notes else "")})
        turn += 1
    if final is None:
        messages.append({"role": "user", "content": "You have used all your rounds. Submit your final law now with "
                                                    "the <final_law> tag."})
        for _ in range(3):
            txt = call(MAX_ROUNDS + 1)
            if law_of(txt):
                note = gate(MAX_ROUNDS + 1, law_of(txt), 0)
                if not note:
                    final = law_of(txt)
                    break
                messages.append({"role": "user", "content": note.strip() + "\nThis is still your last action: reply "
                                                                           "with one <final_law>."})
        if es.reviewed and final:
            es.review(MAX_ROUNDS + 1, final)
    ev = oracle.evaluate_law(final) if final else {"rmsle": float("nan"), "exact_accuracy": 0.0, "error": "no law"}
    res = {"task": f"{domain}__{difficulty}__{law_version}", "domain": domain, "difficulty": difficulty,
           "law_version": law_version, "method": method, "model": model, "final_law": final,
           "ground_truth_law": oracle.get_ground_truth_law_str(), "rmsle": ev.get("rmsle"),
           "exact": ev.get("exact_accuracy"), "eval_error": ev.get("error"), "rounds": turn, "n_experiments": used,
           "tokens": tokens, "messages": messages}
    res.update(es.record())
    return res
