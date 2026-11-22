"""Offline test of the core (no LLM calls): a toy domain where hypotheses are Python laws y = f(x) and experiments are
exact measurements of y = 2 x^1.5."""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epicompass.session import EpiSession  # noqa: E402
from epicompass import explore  # noqa: E402


class ToyDomain:
    error_unit = "relative error"
    misfit_threshold = 0.01

    def _fn(self, artifact):
        ns = {"math": math}
        exec(artifact, ns)
        return ns["discovered_law"]

    def key(self, artifact):
        return re.sub(r"\s+", "", artifact)

    def n_params(self, artifact):
        return 0

    def errors(self, artifact, obs, refit=True):
        f = self._fn(artifact)
        out = []
        for o in obs:
            try:
                p = f(o.inputs["x"])
                out.append(abs(p - o.outputs) / abs(o.outputs))
            except Exception:  # noqa: BLE001
                out.append(float("inf"))
        return out

    def describe_obs(self, o):
        return f"experiment {o.label}"

    def compare(self, artifact, o):
        return f"observed {o.outputs:.4g}, the law gives {self._fn(artifact)(o.inputs['x']):.4g}"


TRUE = lambda x: 2 * x ** 1.5
H_SQ = "def discovered_law(x):\n    return 2 * x ** 2"
H_OK = "def discovered_law(x):\n    return 2 * x ** 1.5"


def test_session_flow():
    s = EpiSession(ToyDomain(), model=None)                   # no model: formula belief, code-only card
    for i, x in enumerate([1.0, 2.0, 4.0], 1):
        s.on_obs(1, {"x": x, "z": 3.0}, TRUE(x), f'{{"x":{x},"z":3.0}}')
    note = s.round_notes(1)
    assert "Open directions" in note and "z (always 3)" in note      # exploration state before any hypothesis
    s.on_hyp(2, "text", H_SQ)
    review = s.review(2, H_SQ, extra="You have 5 experiment turn(s) left.")
    assert "## Hypotheses" in review and "conflict" in review and "Before this submission is scored" in review
    assert "explained by none of your hypotheses" in review
    s.on_hyp(3, "text", H_OK)
    assert s.st.hyps[-1].parent == "H1"                              # the revision is attached under H1
    assert s.final_check(3, H_OK) == ""                              # better than the reviewed law: no regression note
    rec = s.record()
    assert rec["epic_state"]["hyps"][1]["parent"] == "H1"
    assert rec["epic_summary"]["reviewed"]


def test_regression_note():
    s = EpiSession(ToyDomain(), model=None)
    for x in [1.0, 2.0, 3.0]:
        s.on_obs(1, {"x": x}, TRUE(x), f'{{"x":{x}}}')
    s.round_notes(1)
    s.review(2, H_OK)
    note = s.final_check(2, H_SQ)                                    # the revision is worse on the agent's own data
    assert "your revised submission" in note


def test_explore_inputs():
    class O:
        def __init__(self, inputs, label=""):
            self.inputs, self.outputs, self.label = inputs, None, label
    assert explore.inputs_of(O({"a": 1, "b": [1.0, 2.0], "_x": 5})) == {"a": 1.0, "b[0]": 1.0, "b[1]": 2.0}
    assert explore.inputs_of(O("id", 'round 1 #1: {"p": 2}')) == {"p": 2.0}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
