"""Update: the hypothesis check and the belief update as E_t grows.

Hypothesis check: err(h, o), the domain's error of hypothesis h on experiment o (parameters the benchmark would fit
are fitted by the domain); inf = no finite prediction, None = unknown. Cached per hypothesis version.
Belief update (relative credence over the agent's own hypotheses, not a probability of being true): stated by the
backbone itself from the hypotheses and their evidence (belief.py), once per round. Without an LLM it falls back to
  log b(h) = - sum_o min(cap, log((err(h,o) + eta) / (min_g err(g,o) + eta)))  -  (k_h / 2) log n     (normalised)
One update per round (a tool call may run several experiments); the findings compare the belief before and after --
  D1 the favoured hypothesis changed; D2 the working hypothesis lost >= 3x odds to the leader;
  D3 two hypotheses no earlier experiment separated are separated (>= 3x) by one of this round's;
  D4 an observation no hypothesis explains (each misses it by > 5x its own earlier median).
"""
from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional

from .state import INF, Finding, Hyp

OBS_CAP = math.log(1e4)       # at most a 10^4 odds shift from a single observation
SHIFT = math.log(3.0)         # D1/D2: a belief change counts when the round moves the odds >= 3x
SPLIT_FACTOR = 3.0            # D3: former rivals separated when their errors on a new observation differ >= 3x
SEPARATE_FACTOR = 1.5         # ... having been within 1.5x on every earlier observation
OUTLIER_FACTOR = 5.0          # D4 (and S1 outliers): error > 5x the hypothesis' own median error
TOP_REFIT = 2                 # parameters re-profiled only for the working hypothesis and the 2 most credible others


def total(errs):
    return statistics.mean(errs) if errs else INF


def _refit_ids(E):
    ids = {E.current.id} if E.current is not None else set()
    ids |= {hid for hid, _ in sorted(E.cred.items(), key=lambda kv: -kv[1])[:TOP_REFIT]}
    return ids if E.cred else {h.id for h in E.hyps}


def errors(E, h: Hyp, upto: Optional[int] = None, refit: Optional[bool] = None) -> Optional[List[float]]:
    obs = E.obs if upto is None else E.obs[:upto]
    if refit is None:
        refit = h.id in _refit_ids(E)
    k = (h.id, hash(h.artifact), len(obs))
    if k not in E._err:
        try:
            e = E.d.errors(h.artifact, obs, refit=refit)
        except Exception:  # noqa: BLE001 -- an artifact that raises is the agent's failing hypothesis
            e = [INF] * len(obs)
        E._err[k] = None if e is None else [x if (x is not None and x == x) else INF for x in e]
    return E._err[k]


def errors_of(E, artifact, upto=None):
    """Errors of an artifact that is not (or no longer) a node, e.g. the version of a hypothesis the agent replaced."""
    obs = E.obs if upto is None else E.obs[:upto]
    try:
        e = E.d.errors(artifact, obs, refit=True)
    except Exception:  # noqa: BLE001
        e = [INF] * len(obs)
    return None if e is None else [x if (x is not None and x == x) else INF for x in e]


def best_other(E, h: Hyp, upto=None):
    c = [(total(e), g) for g in E.hyps if g.key != h.key for e in [errors(E, g, upto)] if e is not None]
    c = [x for x in c if x[0] < INF]
    return min(c, key=lambda x: x[0]) if c else (INF, None)




def credence(E, upto: Optional[int] = None) -> Dict[str, float]:
    n = len(E.obs) if upto is None else upto
    if getattr(E, "believer", None) is not None:                     # LLM belief update (belief.py)
        b = E.believer(E, errors, total, n)
        return dict(E.cred) if b is None else b
    M = {h.id: errors(E, h, n) for h in E.hyps}
    M = {k: v for k, v in M.items() if v is not None and len(v) == n}
    if not M:
        return {}
    if n == 0:
        return {k: 1.0 / len(M) for k in M}
    logw = {k: 0.0 for k in M}
    for j in range(n):
        col = {k: v[j] for k, v in M.items()}
        fin = [x for x in col.values() if x < INF]
        if not fin:
            continue
        best = min(fin)
        eta = 1e-12 + 1e-6 * best
        for k, x in col.items():
            logw[k] -= OBS_CAP if x == INF else min(OBS_CAP, math.log((x + eta) / (best + eta)))
    for h in E.hyps:
        if h.id in logw:
            try:
                kp = int(E.d.n_params(h.artifact))
            except Exception:  # noqa: BLE001
                kp = 0
            logw[h.id] -= 0.5 * kp * math.log(max(n, 2))
    m = max(logw.values())
    z = sum(math.exp(v - m) for v in logw.values())
    return {k: math.exp(v - m) / z for k, v in logw.items()}




def refresh(E):
    """Belief after a change that brought no observation (e.g. a new hypothesis)."""
    E.cred = credence(E)


def _observe(E, o, update_belief=True) -> List[Finding]:
    n = E.obs.index(o) + 1
    if update_belief:
        E.cred = credence(E, n)
        E.belief_log.append({"obs": o.id, "turn": o.turn, "belief": {k: round(v, 4) for k, v in E.cred.items()}})
    out: List[Finding] = []
    M = {h.id: errors(E, h, n) for h in E.hyps}
    M = {k: v for k, v in M.items() if v is not None and len(v) == n}
    if M and n >= 4:                                                    # D4
        unexpl = [k for k, v in M.items()
                  if not [x for x in v[:-1] if x < INF] or v[-1] == INF
                  or v[-1] > OUTLIER_FACTOR * max(statistics.median([x for x in v[:-1] if x < INF]), 1e-12)]
        if len(unexpl) == len(M) and any(v[-1] < INF for v in M.values()):
            out.append(Finding("D4", o.id, {"errs": {k: M[k][-1] for k in M}, "sig": o.id}))
    ids = list(M)                                                       # D3
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ea, eb = M[a], M[b]
            if n < 2 or any(x == INF for x in ea[:-1] + eb[:-1]):
                continue
            tied = all(max(x, y) <= SEPARATE_FACTOR * max(min(x, y), 1e-12) for x, y in zip(ea[:-1], eb[:-1]))
            if tied and max(ea[-1], eb[-1]) > SPLIT_FACTOR * max(min(ea[-1], eb[-1]), 1e-12):
                win, lose = (a, b) if ea[-1] < eb[-1] else (b, a)
                out.append(Finding("D3", o.id, {"win": win, "lose": lose, "e_win": min(ea[-1], eb[-1]),
                                                "e_lose": max(ea[-1], eb[-1]), "sig": (o.id, win, lose)}))
    return out


def _shift(E, o, before, after) -> List[Finding]:
    out = []
    if len(after) < 2 or len(before) < 2:
        return out
    lead_b, lead_a = max(before, key=before.get), max(after, key=after.get)
    odds = lambda c, a, b: math.log(max(c.get(a, 0), 1e-300) / max(c.get(b, 0), 1e-300))
    if lead_a != lead_b and lead_b in after and odds(after, lead_a, lead_b) - odds(before, lead_a, lead_b) >= SHIFT:
        out.append(Finding("D1", o.id, {"new": lead_a, "old": lead_b, "odds_before": math.exp(odds(before, lead_a, lead_b)),
                                        "odds_after": math.exp(odds(after, lead_a, lead_b)), "sig": o.id}))
    cur = E.current.id if E.current is not None else None
    if cur in after and cur in before and cur != lead_a and not (out and out[-1].detail["old"] == cur):
        d = odds(after, lead_a, cur) - odds(before, lead_a, cur)
        if d >= SHIFT:
            out.append(Finding("D2", o.id, {"cur": cur, "leader": lead_a, "factor": math.exp(d),
                                            "odds_after": math.exp(odds(after, lead_a, cur)), "sig": o.id}))
    return out


def step(E, new_obs) -> List[Finding]:
    """One transition for the observations of one round, in the order they were made."""
    if not new_obs:
        return []
    llm = getattr(E, "believer", None) is not None
    before = dict(E.cred) if llm else credence(E, E.obs.index(new_obs[0]))
    per = []
    for o in new_obs:
        per += _observe(E, o, update_belief=not llm)
    if llm:                                                            # one belief per round
        E.cred = credence(E)
        E.belief_log.append({"obs": new_obs[-1].id, "turn": new_obs[-1].turn,
                             "belief": {k: round(v, 4) for k, v in E.cred.items()}})
    out = []
    for kind in ("D4", "D3"):
        evs = [v for v in per if v.kind == kind]
        if evs:
            v = evs[0]
            if len(evs) > 1:
                v.detail["also"] = [e.subject for e in evs[1:]]
            out.append(v)
    return out + _shift(E, new_obs[-1], before, E.cred)
