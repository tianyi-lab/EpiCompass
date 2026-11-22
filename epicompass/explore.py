"""Exploration update U_{t-1} -> U_t: the recognised directions toward unexplored hypotheses, computed from the agent's
own experiments and hypotheses and shown as "## Open directions" in the epistemic feedback (EPIC_EXPLORE=1). Facts
only; it never suggests a law, a form or a parameter value.
  unexplained  experiments that none of the agent's hypotheses explains (no hypothesis is supported by them)
  untested     experiment inputs the agent has never varied, or varied at only two values
  undecided    the two most believed hypotheses, when none of the agent's experiments separates them
"""
import json

from . import config

EXPLORE = config.EXPLORE
MAX_LISTED = 4
MIN_OBS = 3                     # coverage facts only once a few experiments exist


def _flat(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            if str(k).startswith("_") or k in ("measurement_times", "seed"):
                continue
            out.update(_flat(v, f"{prefix}{k}" if not prefix else f"{prefix}.{k}"))
    elif isinstance(x, (list, tuple)):
        if len(x) <= 3:
            for i, v in enumerate(x):
                out.update(_flat(v, f"{prefix}[{i}]"))
    elif isinstance(x, bool):
        pass
    elif isinstance(x, (int, float)):
        out[prefix or "x"] = float(x)
    return out


def inputs_of(o):
    """The numeric experiment inputs of an observation, flattened ({name: value}), whatever the domain stores."""
    if isinstance(o.inputs, dict):
        return _flat(o.inputs)
    if isinstance(o.outputs, dict) and isinstance(o.outputs.get("_input"), dict):
        return _flat(o.outputs["_input"])
    lab = o.label or ""
    i = lab.find("{")
    if i >= 0:
        try:
            return _flat(json.loads(lab[i:]))
        except Exception:  # noqa: BLE001 -- labels may be clipped
            return {}
    return {}


def untested(st):
    if len(st.obs) < MIN_OBS:
        return [], []
    vals = {}
    for o in st.obs:
        for k, v in inputs_of(o).items():
            vals.setdefault(k, set()).add(round(v, 10))
    never = sorted(k for k, s in vals.items() if len(s) == 1)
    two = sorted(k for k, s in vals.items() if len(s) == 2)
    return [(k, next(iter(vals[k]))) for k in never], two


def unexplained(st):
    from .card import _ev
    if not st.hyps or not st.obs:
        return []
    lab = [{o.id: r for o, r, _ in _ev(st, h)} for h in st.hyps]
    return [o for o in st.obs if all(m.get(o.id) != "support" for m in lab)]


def undecided(st):
    from .card import _ev
    b = {h.id: st.cred.get(h.id, 0.0) for h in st.hyps}
    if len(b) < 2 or not st.obs:
        return None
    a, c = sorted(b, key=b.get, reverse=True)[:2]
    ha, hc = (next(h for h in st.hyps if h.id == x) for x in (a, c))
    la = {o.id: r for o, r, _ in _ev(st, ha)}
    lc = {o.id: r for o, r, _ in _ev(st, hc)}
    if any(la.get(o.id) != lc.get(o.id) for o in st.obs):
        return None
    return a, c


def _fmt(v):
    return f"{v:.4g}"


def lines(st, d):
    """The "## Open directions" section (empty list when there is nothing to report)."""
    out = []
    un = unexplained(st)
    if un:
        names = [d.describe_obs(o) for o in un]
        verb = "is" if len(un) == 1 else "are"
        out.append(f"- {len(un)} of your {len(st.obs)} experiments {verb} explained by none of your hypotheses: "
                   + "; ".join(names[:MAX_LISTED]) + (f"; and {len(names) - MAX_LISTED} more" if len(names) > MAX_LISTED else ""))
    never, two = untested(st)
    if never:
        out.append("- Inputs you have not varied yet (same value in all your experiments): "
                   + ", ".join(f"{k} (always {_fmt(v)})" for k, v in never[:6]) + (" ..." if len(never) > 6 else ""))
    if two:
        out.append("- Inputs you have varied at only two values: " + ", ".join(two[:6]) + (" ..." if len(two) > 6 else ""))
    ud = undecided(st)
    if ud:
        out.append(f"- None of your experiments separates {ud[0]} and {ud[1]}: they agree on every experiment so far.")
    return (["", "## Open directions"] + out) if out else []


def signature(st, d):
    """What changed in U_t (a new card is worth showing only when this changes)."""
    never, two = untested(st)
    ud = undecided(st)
    return (tuple(o.id for o in unexplained(st)), tuple(k for k, _ in never), tuple(two), ud)
