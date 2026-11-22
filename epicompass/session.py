"""EpiSession: the one object a benchmark adapter holds. Benchmark-independent.

    s = EpiSession(domain, model)
    s.on_obs(turn, inputs, outputs, label)     # an experiment the agent ran and its outcome
    s.on_hyp(turn, source, artifact)           # an executable hypothesis the agent wrote
    feedback = s.round_notes(turn)             # belief + exploration update for the round, then epistemic feedback ("" if none)
    review = s.review(turn, artifact)          # self-verification at the first submission ("" if nothing to report)
    regress = s.final_check(turn, artifact)    # at the second submission: "" unless the revision is worse than before

Hypothesis selection by epistemic gain lives in selection.py and is called by the adapter at hypothesis steps.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from . import checks as U
from . import config
from . import construction as C
from . import update as T
from .render import HEADER, _fmt, footer, render
from .state import EpistemicState, Finding

MAIN_MARGIN = 0.05            # a new main hypothesis must exceed the old one's belief by more than this


class EpiSession:
    def __init__(self, domain, model=None, max_notes_per_episode=None):
        self.d = domain
        self.model = model
        self.st = EpistemicState(domain)
        if model and config.BELIEF == "llm":
            from .belief import Believer
            self.st.believer = Believer(model, config.EFFORT)
        if model:
            from .card import CardWriter
            self.st.card_writer = CardWriter(model, config.EFFORT)
        self._top_prev = None
        self.shown = set()
        self.reviewed = False
        self.final_checked = False
        self.max_notes = config.MAX_NOTES if max_notes_per_episode is None else max_notes_per_episode
        self.n_notes = 0
        self.counts: Counter = Counter()
        self.events: List[Dict[str, Any]] = []
        self._new_obs: List[Any] = []
        self._new_hyp = False
        self._review_artifact = None
        self._tree_sig = None
        self._u_sig = None

    @property
    def current(self):
        return self.st.current

    # -- construction --------------------------------------------------------------------------------------------------
    def on_obs(self, turn, inputs, outputs, label=""):
        o = C.add_observation(self.st, turn, inputs, outputs, label)
        self._new_obs.append(o)
        return o

    def on_hyp(self, turn, source, artifact):
        h, status = C.add_hypothesis(self.st, turn, source, artifact)
        self._new_hyp = True
        self.events.append({"turn": turn, "event": "hyp", "id": h.id, "status": status, "source": source})
        return h

    # -- update --------------------------------------------------------------------------------------------------------
    def _update(self):
        found = T.step(self.st, self._new_obs)
        if not self._new_obs and self._new_hyp:
            T.refresh(self.st)
        self._new_obs, self._new_hyp = [], False
        return found

    def _belief(self):
        return {k: round(v, 4) for k, v in self.st.cred.items()}

    def _main_change(self):
        """M1: the hypothesis with the highest belief is no longer the main one (or, the first time a belief compares
        hypotheses, it is not the one the agent wrote last)."""
        from .card import top_hyp
        top = top_hyp(self.st)
        if top is None:
            return []
        b = self.st.cred
        prev = self._top_prev
        if prev is not None and prev in b and b[top] <= b[prev] + MAIN_MARGIN:
            top = prev                              # no clear change: the main hypothesis stays
        self._top_prev = top
        self.st.main_id = top
        latest = self.st.current.id if self.st.current is not None else None
        sig = ("main", top, len(self.st.obs), len(self.st.hyps))
        if prev is None:
            if latest is None or latest == top:
                return []
            return [Finding("M1", top, {"top": top, "b": b[top], "latest": latest, "latest_b": b.get(latest, 0.0),
                                        "sig": sig})]
        if prev == top:
            return []
        return [Finding("M1", top, {"top": top, "b": b[top], "prev": prev, "prev_b": b.get(prev, 0.0), "sig": sig})]

    def _fresh(self, fs):
        out = [f for f in fs if f.sig not in self.shown]
        self.shown |= {f.sig for f in out}
        return out

    def _open_directions(self):
        """(lines of the "## Open directions" section, whether U_t changed since it was last shown)."""
        from . import explore as X
        if not config.EXPLORE:
            return [], False
        try:
            self._u_cur = X.signature(self.st, self.d)
            ul = X.lines(self.st, self.d)
        except Exception as e:  # noqa: BLE001 -- never break the episode
            self.counts["explore_error"] += 1
            self.events.append({"event": "explore_error", "error": repr(e)[:300]})
            return [], False
        new = bool(ul) and self._u_cur != self._u_sig
        if new:
            self.events.append({"event": "open_directions", "n_obs": len(self.st.obs), "lines": ul[2:]})
        return ul, new

    # -- epistemic feedback --------------------------------------------------------------------------------------------
    def round_notes(self, turn):
        """Belief and exploration update for the round's new experiments / hypotheses, then the feedback ("" if
        nothing new or the per-episode cap is reached)."""
        if config.SHOW == "none":                       # ablation: the state is kept (selection uses it), not shown
            self._update()
            self._main_change()
            return ""
        if config.SHOW == "tree":                       # ablation: only the revision tree is shown
            self._update()
            self._main_change()
            return self._tree_notes(turn)
        fs = self._fresh(self._update() + self._main_change() + U.query(self.st, self.current, at_submit=False))
        for f in fs:
            self.counts[f"round_{f.kind}"] += 1
        ulines, unew = self._open_directions()
        if (not fs and not unew) or self.n_notes >= self.max_notes:
            return ""
        self.n_notes += 1
        txt = render(self.st, fs, self.d) if fs else ""
        if ulines:
            from .card import HEAD
            txt = (txt if txt.strip() else "\n" + HEAD) + "\n" + "\n".join(ulines)
            self._u_sig = self._u_cur
            self.counts["open_directions_shown"] += 1
        self.events.append({"turn": turn, "event": "notes", "kinds": [f.kind for f in fs], "chars": len(txt),
                            "text": txt[:2000], "belief": self._belief()})
        return txt

    def _tree_notes(self, turn, force=False):
        """EPIC_SHOW=tree: the agent's hypotheses as a revision tree only (no evidence, no belief, no findings)."""
        sig = tuple((h.id, hash(h.artifact)) for h in self.st.hyps)
        if not self.st.hyps or (not force and (sig == self._tree_sig or self.n_notes >= self.max_notes)):
            return ""
        self._tree_sig = sig
        from .card import hyp_text
        cur = self.st.current.id if self.st.current is not None else None
        byid = {h.id: h for h in self.st.hyps}
        kids = {}
        for h in self.st.hyps:
            kids.setdefault(h.parent if h.parent in byid else None, []).append(h)
        lines = ["", "[Your hypotheses so far, as a revision tree: each child is a revision of its parent]"]

        def walk(pid, dep):
            for h in sorted(kids.get(pid, []), key=lambda g: (g.id != cur, g.id)):
                ind = "    " * dep
                lines.append(f"{ind}{'└─ ' if dep else '- '}{h.id} {'[Main]' if h.id == cur else '[Alternative]'}")
                lines.extend(ind + x for x in hyp_text(self.d, h))
                walk(h.id, dep + 1)
        walk(None, 0)
        self.n_notes += 1
        txt = "\n".join(lines)
        self.events.append({"turn": turn, "event": "notes", "kinds": ["tree"], "chars": len(txt), "text": txt[:2000]})
        return txt

    # -- self-verification ---------------------------------------------------------------------------------------------
    def review(self, turn, artifact, extra=""):
        """At the first submission: the complete epistemic tree and every finding about the submission; the agent may
        revise, experiment further, or confirm, and its next submission is final. Later calls only log the submission."""
        if not config.REVIEW:                           # ablation: no self-verification, the submission stands
            self._update()
            h = self.on_hyp(turn, "submission", artifact)
            self.events.append({"turn": turn, "event": "final_submission", "hyp": h.id, "review": "off"})
            self.reviewed = self.final_checked = True
            return ""
        if config.SHOW == "tree":                       # ablation: the verification shows only the revision tree
            self._update()
            h = self.on_hyp(turn, "submission", artifact)
            if self.reviewed:
                self.events.append({"turn": turn, "event": "final_submission", "hyp": h.id, "review": "tree"})
                return ""
            self.reviewed = True
            txt = self._tree_notes(turn, force=True) + "\n\n" + footer(extra)
            self.events.append({"turn": turn, "event": "review", "hyp": h.id, "kinds": ["tree"], "chars": len(txt),
                                "text": txt[:2500]})
            return txt
        self._update()
        h = self.on_hyp(turn, "submission", artifact)
        T.refresh(self.st)
        fs = self._main_change() + U.query(self.st, h, at_submit=True)
        for f in fs:
            self.counts[f"submit_{f.kind}"] += 1
        if self.reviewed:
            self.events.append({"turn": turn, "event": "final_submission", "hyp": h.id, "kinds": [f.kind for f in fs],
                                "belief": self._belief()})
            return ""
        self.reviewed = True
        self._review_artifact = h.artifact
        txt = render(self.st, fs, self.d, at_submit=True, extra=extra)
        ulines, _ = self._open_directions()
        if ulines and txt:
            cut = txt.rfind("\n\nBefore this submission is scored")
            txt = (txt[:cut] + "\n" + "\n".join(ulines) + txt[cut:]) if cut >= 0 else txt + "\n" + "\n".join(ulines)
            self.counts["open_directions_review"] += 1
        self.events.append({"turn": turn, "event": "review", "hyp": h.id, "kinds": [f.kind for f in fs],
                            "chars": len(txt), "text": txt[:2500], "belief": self._belief()})
        self.shown |= {f.sig for f in fs}
        return txt

    def final_check(self, turn, artifact):
        """At the second submission: a note only if the revision is worse on the agent's own experiments than the
        submission it revised."""
        if self.final_checked or self._review_artifact is None:
            return ""
        self.final_checked = True
        self._update()
        new = C.normalise(artifact)
        if new == self._review_artifact or not self.st.obs:
            return ""
        r = U.regression(self.st, self._review_artifact, new)
        if r is None:
            return ""
        m_old, m_new, f_old, f_new, regressed = r
        self.events.append({"turn": turn, "event": "final_check", "regressed": regressed, "old": m_old, "new": m_new})
        if not regressed:
            return ""
        self.counts["regression"] += 1
        what = f"gives no finite prediction for {f_new} of them" if f_new > f_old else f"has mean error {_fmt(m_new)}"
        return (f"\n{HEADER}\n- On all {len(self.st.obs)} of your experiments, your revised submission {what}; the "
                f"submission you revised had mean error {_fmt(m_old)}. Submit whichever you want scored (either one, "
                f"or another); your next submission is final.")

    # -- records -------------------------------------------------------------------------------------------------------
    def summary(self):
        s = {"method": "epicompass", "n_obs": len(self.st.obs), "n_hyps": len(self.st.hyps), "n_notes": self.n_notes,
             "reviewed": self.reviewed, "final_checked": self.final_checked, "counts": dict(self.counts)}
        for name, obj in (("belief", self.st.believer), ("card", self.st.card_writer)):
            if obj is not None:
                s.update({f"{name}_calls": obj.calls, f"{name}_tokens": obj.tokens, f"{name}_errors": obj.errors})
        return s

    def record(self):
        tot = lambda h: (lambda e: None if e is None else T.total(e))(T.errors(self.st, h))
        return {"epic_summary": self.summary(), "epic_events": self.events, "epic_state": self.st.to_dict(total=tot)}
