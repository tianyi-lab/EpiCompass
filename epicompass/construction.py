"""Construction: the agent-environment trajectory -> E_t. Two event types, each mapped to one update:

  observation  an experiment and its outcome, logged by the harness  -> D_t := D_t + {o}
  hypothesis   an executable hypothesis the agent wrote              -> H_t: merged by the domain's key (a new version
                                                                        replaces the old artifact); a new hypothesis is
                                                                        attached under the one it most resembles
The agent writes nothing for the state: everything is recorded from its own actions.
"""
from __future__ import annotations

from .state import Hyp, Obs


def normalise(artifact: str) -> str:
    return "\n".join(l.rstrip() for l in (artifact or "").strip().splitlines() if l.strip())


def add_observation(E, turn, inputs, outputs, label=""):
    o = Obs(f"O{len(E.obs) + 1}", turn, inputs, outputs, label)
    E.obs.append(o)
    return o


def add_hypothesis(E, turn, source, artifact):
    """Returns (hyp, status): "new" | "same" (the latest hypothesis again) | "returned" (an earlier one, after a
    different one was written in between)."""
    norm = normalise(artifact)
    try:
        key = E.d.key(norm)
    except Exception:  # noqa: BLE001 -- an artifact that does not parse is still the agent's hypothesis
        key = ("unparsed", hash(norm))
    latest = max(E.hyps, key=lambda x: x.last_turn) if E.hyps else None
    for h in E.hyps:
        if h.key == key:
            status = "same" if h is latest else "returned"
            if status == "returned":
                E.returns.append({"hyp": h.id, "turn": turn, "last_turn": h.last_turn, "artifact_then": h.artifact,
                                  "n_obs_then": sum(1 for o in E.obs if o.turn <= h.last_turn), "n_obs_now": len(E.obs),
                                  "checked": False})
            h.artifact, h.last_turn = norm, turn
            h.seen.append(turn)
            if source not in h.sources:
                h.sources.append(source)
            E.current = h
            return h, status
    h = Hyp(f"H{len(E.hyps) + 1}", turn, source, norm, key, turn, [turn], [source])
    if E.hyps:
        try:
            from .card import infer_parent
            h.parent = infer_parent(E, h)
        except Exception:  # noqa: BLE001 -- the tree edge is never allowed to break the episode
            h.parent = None
    E.hyps.append(h)
    E.current = h
    return h, "new"
