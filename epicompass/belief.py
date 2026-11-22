"""Belief update over the explored hypotheses: b_t = BELIEF(H_t, V_t). The backbone itself reads the hypotheses and
their evidence (each experiment's prediction error and its support / conflict / neutral label) and states a
distribution over them. Same model and reasoning effort as the agent. Called once per update and cached; on a failed
call the previous belief is kept."""
from __future__ import annotations

from .llm import call, parse_json
from .state import INF

MAX_HYPS = 10        # the latest hypothesis + the best-fitting others; the rest get belief 0
MAX_OBS = 30         # the most recent experiments are listed one by one; the mean errors are over all of them
ARTIFACT_CHARS = 1200

PROMPT = """A scientist is studying an unknown system. Below are the candidate hypotheses they wrote and, as evidence, \
how each of the scientist's own experiments bears on each hypothesis: its prediction error ({unit}) and whether the \
experiment supports it, conflicts with it, or gives no evidence either way ("fails" = no finite prediction).

Based on this evidence, state your belief that each hypothesis is the correct one among these candidates: a number in \
[0, 1] per hypothesis, summing to 1. Weigh how well each fits all of the experiments, any experiment it fails or \
misses badly, and that a hypothesis with more free parameters fits more easily.

HYPOTHESES:
{hyps}

EVIDENCE (rows = experiments, columns = hypotheses):
{table}

Return JSON only: {{"beliefs": {{{ids}}}}}"""

WORD = {"support": "supports", "conflict": "conflicts", "none": "no evidence"}


def _fmt(x):
    return "fails" if x == INF else f"{x:.3g}"


class Believer:
    def __init__(self, model, effort=None):
        self.model, self.effort = model, effort
        self.calls, self.tokens, self.errors = 0, 0, 0
        self._cache = {}

    def __call__(self, E, errors, total, n):
        """errors(E, h, n) -> per-observation errors of h on the first n observations (or None)."""
        from .checks import evidence
        M = {h.id: errors(E, h, n) for h in E.hyps}
        M = {k: v for k, v in M.items() if v is not None and len(v) == n}
        if not M:
            return {}
        if n == 0 or len(M) == 1:
            return {k: 1.0 / len(M) for k in M}
        cur = E.current.id if E.current is not None and E.current.id in M else None
        ranked = sorted(M, key=lambda k: total(M[k]))
        keep = ([cur] if cur else []) + [k for k in ranked if k != cur][:MAX_HYPS - (1 if cur else 0)]
        hyps = [h for h in E.hyps if h.id in keep]
        ck = (tuple((h.id, hash(h.artifact)) for h in hyps), n)
        if ck in self._cache:
            return self._cache[ck]
        lab = {h.id: [r for _, r, _ in evidence(E, h)][:n] for h in hyps}
        hs = "\n\n".join(f"{h.id} ({self._np(E, h)} free parameters, mean error {_fmt(total(M[h.id]))}):\n"
                          f"{h.artifact[:ARTIFACT_CHARS]}" for h in hyps)
        obs = list(range(n))[-MAX_OBS:]
        head = "experiment | " + " | ".join(h.id for h in hyps)
        rows = [f"{E.obs[j].id} {E.d.describe_obs(E.obs[j])[:120]} | " +
                " | ".join(f"{_fmt(M[h.id][j])} ({WORD.get(lab[h.id][j], 'no evidence')})" for h in hyps) for j in obs]
        if n > MAX_OBS:
            rows.insert(0, f"(the {n - MAX_OBS} earliest experiments are not listed; the mean errors above include them)")
        prompt = PROMPT.format(unit=getattr(E.d, "error_unit", "") or "relative", hyps=hs,
                               table="\n".join([head] + rows), ids=", ".join(f'"{h.id}": <belief>' for h in hyps))
        txt, usage = call(self.model, prompt, self.effort)
        self.calls += 1
        self.tokens += int(usage.get("total_tokens") or 0)
        b = self._parse(txt, [h.id for h in hyps])
        if b is None:
            self.errors += 1
            return None
        out = {k: b.get(k, 0.0) for k in M}
        self._cache[ck] = out
        return out

    @staticmethod
    def _np(E, h):
        try:
            return int(E.d.n_params(h.artifact))
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _parse(txt, ids):
        d = parse_json(txt)
        try:
            d = d.get("beliefs", d)
            v = {k: max(0.0, float(d[k])) for k in ids if k in d}
        except Exception:  # noqa: BLE001
            return None
        z = sum(v.values())
        if not v or z <= 0:
            return None
        return {k: x / z for k, x in v.items()}
