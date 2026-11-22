"""Baselines that re-implement the core idea of related work on the same harness (EPIC_METHOD=popper | hsearch).

popper   Popper (Huang et al., 2025): sequential falsification. For the agent's current hypothesis H, the backbone
         designs a falsification experiment (relevance-checked), the harness runs it, and H's prediction is checked on
         the outcome; each pass doubles H's evidence E, a failure falsifies H. A submission whose hypothesis has not
         reached E >= 1/alpha (or was falsified) is held back once, with the test reports.
hsearch  Hypothesis Search (Wang et al., 2024): at a hypothesis step K replies are sampled and the backbone picks the
         one it finds most promising from the conversation and the replies (selection.self_choose). Nothing is
         executed for it and nothing is shown to the agent.
"""
from __future__ import annotations

import json

from . import config
from .card import hyp_text, _ev
from .llm import call, parse_json
from .session import EpiSession

HEAD = "[Research state - computed only from your own experiments and models]"


class PopperSession(EpiSession):
    DESIGN = """A scientist holds the hypothesis {hid} about an unknown system:
{hyp}

Experiments run so far (inputs, in the exact format the system accepts):
{examples}

Design ONE falsification experiment for {hid}: an experiment where, if {hid} is wrong, its prediction is most likely \
to be wrong. Prefer inputs unlike those already tested. If the hypothesis is false, the experiment's outcome should \
contradict its prediction.

Return JSON only: {{"input": <ONE experiment input in exactly the format of the examples>, "rationale": "<one sentence>"}}"""
    RELEVANCE = """Hypothesis:
{hyp}

Proposed falsification experiment: {inp}
Rationale: {why}

If the hypothesis' prediction for this experiment turns out wrong, how strongly does that show the hypothesis is \
wrong (rather than being uninformative)? Return JSON only: {{"score": <number in [0, 1]>}}"""
    ALPHA = 0.125          # validated when E >= 1/alpha = 8, i.e. three passed falsification tests
    TAU = 0.8              # relevance threshold
    MAX_TESTS = 6          # per episode

    def __init__(self, domain, model=None, **kw):
        super().__init__(domain, None, **kw)
        self.model = model
        self.E, self.falsified, self.log, self.n_tests = {}, {}, [], 0
        self.io = None          # (examples() -> list of input JSON strings, run(turn, input) -> (obs | None, text))
        self._last_tested = None

    def _design(self, h):
        ex = self.io[0]() if self.io else []
        for _ in range(2):                                  # one redesign if the relevance check fails
            try:
                txt, _ = call(self.model, self.DESIGN.format(hid=h.id, hyp="\n".join(hyp_text(self.d, h)),
                                                             examples="\n".join(ex[-8:]) or "(none yet)"),
                              config.EFFORT)
                d = parse_json(txt) or {}
                inp, why = d.get("input"), str(d.get("rationale", ""))
                r, _ = call(self.model, self.RELEVANCE.format(hyp="\n".join(hyp_text(self.d, h)),
                                                              inp=json.dumps(inp)[:600], why=why[:300]),
                            config.EFFORT)
                score = float((parse_json(r) or {}).get("score", 0))
            except Exception as e:  # noqa: BLE001
                self.events.append({"event": "popper_error", "stage": "design", "error": f"{type(e).__name__}: {str(e)[:200]}"})
                return None, None
            if score >= self.TAU:
                return inp, why
        return None, None

    def _open(self, h):
        return (h is not None and self.io is not None and self.n_tests < self.MAX_TESTS and h.id not in self.falsified
                and self.E.get(h.id, 1.0) < 1 / self.ALPHA)

    def round_notes(self, turn):
        self._update()
        h = self.st.current
        if not self._open(h):
            return ""
        return self._test(turn, h)

    def _test(self, turn, h):
        """One falsification test of h: design (relevance-checked), run, check h's prediction. Returns the report."""
        inp, why = self._design(h)
        if inp is None:
            return ""
        try:
            o, out_txt = self.io[1](turn, inp)
        except Exception as e:  # noqa: BLE001
            self.events.append({"turn": turn, "event": "popper_error", "stage": "run", "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return ""
        if o is None:
            return ""
        self.n_tests += 1
        self._update()
        lab = {ob.id: r for ob, r, _ in _ev(self.st, h)}.get(o.id, "none")
        passed = lab != "conflict"
        if passed:
            self.E[h.id] = self.E.get(h.id, 1.0) * 2.0
        else:
            self.falsified[h.id] = o.id
        try:
            cmp_ = self.d.compare(h.artifact, o)
        except Exception:  # noqa: BLE001
            cmp_ = ""
        status = (f"PASSED - evidence for {h.id} is now {self.E[h.id]:.0f} (validated at {1 / self.ALPHA:.0f})" if passed
                  else f"FAILED - {h.id} is falsified by this experiment")
        txt = "\n".join(["", HEAD, "", f"## Falsification test of {h.id}",
                         f"- experiment (run for you): {json.dumps(inp)[:300]}", f"- why: {why[:300]}",
                         f"- result: {out_txt[:400]}"] + ([f"- comparison: {cmp_}"] if cmp_ else []) + [f"- {status}"])
        self.log.append({"turn": turn, "hyp": h.id, "input": inp, "passed": passed, "E": self.E.get(h.id, 0.0)})
        self.n_notes += 1
        self.events.append({"turn": turn, "event": "notes", "kinds": ["popper"], "text": txt[:2500], **self.log[-1]})
        return txt

    def review(self, turn, artifact, extra=""):
        h = self.on_hyp(turn, "submission", artifact)
        self._update()
        if self.reviewed:
            self.events.append({"turn": turn, "event": "final_submission", "hyp": h.id})
            return ""
        self.reviewed = True
        reports = []
        while self._open(h):                                # Popper: validate the submitted hypothesis now
            r = self._test(turn, h)
            if not r:
                break
            reports.append(r.replace(HEAD, "").strip())
        e = self.E.get(h.id, 1.0)
        if h.id in self.falsified:
            msg = f"Your submission {h.id} was falsified by a falsification test (experiment {self.falsified[h.id]})."
        elif e < 1 / self.ALPHA:
            msg = (f"Your submission {h.id} is not yet validated: its falsification evidence is {e:.0f}, below the "
                   f"threshold {1 / self.ALPHA:.0f} (three passed falsification tests).")
        else:
            self.events.append({"turn": turn, "event": "review", "hyp": h.id, "validated": True})
            return ""
        txt = "\n".join(["", HEAD] + reports + ["", f"- {msg}", "Before this submission is scored you may revise it, investigate further "
                         f"if you have turns left, or submit it unchanged. Your next submission is final. {extra}".strip()])
        self.events.append({"turn": turn, "event": "review", "hyp": h.id, "validated": False, "text": txt})
        return txt

    def final_check(self, turn, artifact):
        return ""

    def summary(self):
        s = super().summary()
        s.update(method="popper", n_tests=self.n_tests, validated=[k for k, v in self.E.items() if v >= 1 / self.ALPHA],
                 falsified=list(self.falsified))
        return s



class HSearchSession(EpiSession):
    """Hypothesis Search: only the selection events (recorded by the adapter) are kept; no state, no feedback."""

    def __init__(self, domain, model=None, **kw):
        super().__init__(domain, None, **kw)

    def on_obs(self, *a, **k):
        return None

    def on_hyp(self, *a, **k):
        return None

    def round_notes(self, turn):
        return ""

    def review(self, turn, artifact, extra=""):
        self.events.append({"turn": turn, "event": "final_submission" if self.reviewed else "review"})
        self.reviewed = True
        return ""

    def final_check(self, turn, artifact):
        return ""

    def record(self):
        n = sum(e.get("event") == "select" for e in self.events)
        return {"epic_summary": {"method": "hsearch", "selects": n}, "epic_events": self.events, "epic_state": {}}


def make_session(domain, model, method=None):
    """The session for a method: epicompass (default) | popper | hsearch | react (no component)."""
    method = method or config.METHOD
    if method == "popper":
        return PopperSession(domain, model)
    if method == "hsearch":
        return HSearchSession(domain, model)
    if method == "react":
        return HSearchSession(domain, None)
    return EpiSession(domain, model)
