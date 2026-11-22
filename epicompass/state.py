"""The epistemic state E_t = (D_t, H_t, b_t, U_t), as data. It never sees the ground truth.

  D_t  observations   every experiment the agent ran and its outcome, logged by the harness
  H_t  hypotheses     executable hypotheses the agent itself wrote, one node per distinct hypothesis, linked into the
                      epistemic tree by `parent` (the hypothesis each one revises)
  b_t  belief         relative credence over H_t given D_t (over the agent's own hypotheses, not a truth probability)
  U_t  exploration    computed on demand from D_t and H_t (explore.py)

The hypothesis-evidence relations (support / conflict / neutral of each experiment for each hypothesis) are computed
from the errors the domain reports and cached here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

INF = float("inf")


@dataclass
class Obs:
    id: str
    turn: int
    inputs: Any
    outputs: Any
    label: str = ""


@dataclass
class Hyp:
    id: str
    turn: int
    source: str
    artifact: str
    key: Any
    last_turn: int = 0
    seen: List[int] = field(default_factory=list)      # turns in which this hypothesis was (re)written
    sources: List[str] = field(default_factory=list)   # how the agent produced it (code, fit, submission, ...)
    parent: Optional[str] = None                       # the hypothesis it revises (None: a new branch of the tree)


def freeze(v):
    """Hashable form of nested containers."""
    if isinstance(v, (set, frozenset)):
        return tuple(sorted((freeze(x) for x in v), key=repr))
    if isinstance(v, (list, tuple)):
        return tuple(freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted(((k, freeze(x)) for k, x in v.items()), key=repr))
    return v


@dataclass
class Finding:
    """One fact computed from the state, reported to the agent at most once (see checks.py and update.py)."""
    kind: str
    subject: str               # hypothesis or observation id
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def sig(self):
        return (self.kind, self.subject, freeze(self.detail.get("sig", "")))


class EpistemicState:
    def __init__(self, domain):
        self.d = domain
        self.obs: List[Obs] = []
        self.hyps: List[Hyp] = []
        self.current: Optional[Hyp] = None           # the hypothesis the agent wrote last
        self.main_id: Optional[str] = None           # the main hypothesis (highest belief)
        self.cred: Dict[str, float] = {}              # belief b_t after the last update
        self.believer = None                          # LLM belief update (belief.py); None = formula
        self.card_writer = None                       # LLM writing of the feedback (card.py); None = code only
        self.belief_log: List[Dict[str, Any]] = []   # belief after each update
        self.returns: List[Dict[str, Any]] = []      # the agent went back to an earlier hypothesis
        self._err: Dict[tuple, Optional[List[float]]] = {}

    def hyp(self, hid) -> Hyp:
        return next(h for h in self.hyps if h.id == hid)

    def to_dict(self, total=None):
        return {"obs": [{"id": o.id, "turn": o.turn, "label": o.label} for o in self.obs],
                "hyps": [{"id": h.id, "turn": h.turn, "last_turn": h.last_turn, "source": h.source,
                          "sources": h.sources, "seen": h.seen, "parent": h.parent, "artifact": h.artifact,
                          "total_error": total(h) if total else None, "belief": self.cred.get(h.id)} for h in self.hyps],
                "belief_log": self.belief_log, "returns": self.returns}
