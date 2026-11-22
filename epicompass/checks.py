"""Hypothesis-evidence relations and the checks reported to the agent. Every finding is relative to the agent's own
experiments and hypotheses: the state never says what is true, and the agent is never blocked.

Evidence relation v(h, d) in {support, conflict, neutral} for each hypothesis h and experiment d (evidence()):
  exact observations (domain.misfit_threshold set): support iff the error is within the threshold, else conflict
  noisy observations: conflict when h gives no finite prediction, misses d by > 5x its own median error, or is >= 2x
  worse than another hypothesis on d; support when h is within 1.5x of the best hypothesis on d and some other
  hypothesis is >= 2x worse there (d separates in its favour); otherwise neutral

Checks on the main hypothesis h
  S1  h gives no finite prediction on some experiments / misses them beyond the exact-data threshold / has outlier
      experiments (> 5x its own median error) / is >= 2x worse on all experiments than another hypothesis
  S3  h is an earlier hypothesis that was already >= 2x worse than another when it was last written, with no new
      experiment since
  R1  rival hypotheses that fit about equally (within 1.5x) and that no experiment separates
  S0  (at submission) no experiment supports the submitted hypothesis
Regression (self-verification): a revised submission that is >= 2x worse on the agent's experiments than the one it
revised, or fails where that one did not.
"""
from __future__ import annotations

import statistics
from typing import List, Optional

from . import update as T
from .state import INF, Finding, Hyp

DOMINATED_FACTOR = 2.0        # S1/S3/regression: >= 2x worse on the same experiments
RIVAL_FACTOR = 1.5            # R1 / support: within 1.5x
SEPARATE_FACTOR = T.SEPARATE_FACTOR


def evidence(E, h):
    """[(obs, 'support' | 'conflict' | 'none', error)] for hypothesis h on every experiment."""
    errs = T.errors(E, h, refit=True)
    if errs is None:
        return [(o, "none", None) for o in E.obs]
    thr = getattr(E.d, "misfit_threshold", None)
    if thr is not None:
        return [(o, "conflict" if (e == INF or e > thr) else "support", e) for o, e in zip(E.obs, errs)]
    fin = [e for e in errs if e < INF]
    med = statistics.median(fin) if fin else INF
    others = [T.errors(E, g, refit=True) for g in E.hyps if g.id != h.id]
    others = [x for x in others if x is not None and len(x) == len(errs)]
    out = []
    for j, (o, e) in enumerate(zip(E.obs, errs)):
        col = [x[j] for x in others if x[j] < INF]
        best = min(col + ([e] if e < INF else []), default=INF)
        if e == INF or (med > 0 and e > T.OUTLIER_FACTOR * med) or (col and e >= DOMINATED_FACTOR * min(col)):
            out.append((o, "conflict", e))
        elif e <= RIVAL_FACTOR * best and any(x[j] >= DOMINATED_FACTOR * max(e, 1e-300) for x in others):
            out.append((o, "support", e))
        else:
            out.append((o, "none", e))
    return out


def s1(E, h) -> List[Finding]:
    out = []
    errs = T.errors(E, h, refit=True)
    if errs is None:
        return out
    bad = [o for o, e in zip(E.obs, errs) if e == INF]
    if bad:
        return [Finding("S1", h.id, {"what": "fails", "obs": [o.id for o in bad], "n": len(E.obs), "sig": ("fails", len(bad))})]
    tot = T.total(errs)
    thr = getattr(E.d, "misfit_threshold", None)
    if thr is not None and tot > thr:
        worst = max(zip(E.obs, errs), key=lambda x: x[1])
        out.append(Finding("S1", h.id, {"what": "misfit", "total": tot, "threshold": thr, "worst": worst[0].id,
                                        "worst_err": worst[1], "sig": ("misfit",)}))
    med = statistics.median(errs)
    outl = sorted([(o, e) for o, e in zip(E.obs, errs) if med > 0 and e > T.OUTLIER_FACTOR * med], key=lambda x: -x[1])
    if outl:
        out.append(Finding("S1", h.id, {"what": "outliers", "median": med, "obs": [(o.id, e) for o, e in outl[:3]],
                                        "sig": ("outl", outl[0][0].id)}))
    bt, g = T.best_other(E, h)
    if g is not None and bt * DOMINATED_FACTOR <= tot:
        out.append(Finding("S1", h.id, {"what": "dominated", "total": tot, "other": g.id, "other_total": bt,
                                        "sig": ("dom", g.id)}))
    return out


def s3(E) -> List[Finding]:
    out = []
    for r in E.returns:
        if r["checked"]:
            continue
        r["checked"] = True
        n = r["n_obs_then"]
        if n == 0 or n != r["n_obs_now"]:
            continue
        e_then = T.errors_of(E, r["artifact_then"], n)
        if e_then is None:
            continue
        tot = T.total(e_then)
        bt, g = T.best_other(E, E.hyp(r["hyp"]), n)
        if g is not None and bt * DOMINATED_FACTOR <= tot:
            out.append(Finding("S3", r["hyp"], {"other": g.id, "total": tot, "other_total": bt, "last_turn": r["last_turn"],
                                                "sig": (r["hyp"], g.id, r["turn"])}))
    return out


def r1(E, h) -> List[Finding]:
    errs = T.errors(E, h)
    if errs is None or T.total(errs) == INF:
        return []
    tot, out = T.total(errs), []
    for g in E.hyps:
        if g.key == h.key:
            continue
        ge = T.errors(E, g)
        if ge is None:
            continue
        gt = T.total(ge)
        if gt == INF or max(gt, tot) > RIVAL_FACTOR * max(min(gt, tot), 1e-12):
            continue
        if any(max(a, b) > SEPARATE_FACTOR * max(min(a, b), 1e-12) for a, b in zip(errs, ge)):
            continue                     # some experiment separates them (S1 covers the loser)
        try:
            disc = E.d.discriminate(h.artifact, g.artifact, E.obs)[:2]
        except Exception:  # noqa: BLE001 -- optional in a domain
            disc = []
        out.append(Finding("R1", h.id, {"rival": g.id, "total": tot, "rival_total": gt, "disc": disc, "sig": g.id}))
    return out


def query(E, h: Optional[Hyp], at_submit=False) -> List[Finding]:
    """All checks for the hypothesis h (the main hypothesis in a round, the submission at the first submission)."""
    f: List[Finding] = []
    if h is not None and E.obs:
        f += s1(E, h)
    f += s3(E)
    if h is not None and E.obs:
        f += r1(E, h)
    if at_submit and h is not None and not any(r == "support" for _, r, _ in evidence(E, h)):
        f.append(Finding("S0", h.id, {"n": len(E.obs), "sig": ("noev", len(E.obs))}))
    return f


def regression(E, old_artifact, new_artifact):
    """(m_old, m_new, n_fail_old, n_fail_new, regressed) of the revised submission against the one it revised."""
    e_old, e_new = T.errors_of(E, old_artifact), T.errors_of(E, new_artifact)
    if e_old is None or e_new is None:
        return None
    m_old, m_new = T.total(e_old), T.total(e_new)
    f_old, f_new = sum(x == INF for x in e_old), sum(x == INF for x in e_new)
    return m_old, m_new, f_old, f_new, (f_new > f_old) or (m_old < INF and m_new >= DOMINATED_FACTOR * m_old)
