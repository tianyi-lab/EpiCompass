"""Hypothesis exploration with epistemic gain. At a hypothesis step the agent's reply is sampled K times from the same
context (h_t^(1..K) ~ pi(. | tau, s)); for every candidate the code computes how the agent's own experiments bear on
it and how it relates to the epistemic tree (a revision of which hypothesis, which experiments it fixes or breaks,
which experiments no existing hypothesis explains but it does), and the backbone selects the candidate with the
highest anticipated epistemic gain: support from the evidence and a promising direction. The selected reply replaces
the proposal in the trajectory.

The Hypothesis Search baseline (Wang et al., 2024) uses `self_choose`: the backbone picks the reply it finds most
promising from the conversation alone (no execution, no evidence, no state).
"""
from __future__ import annotations

from . import config
from . import construction as C
from .card import code_core, edge_lines, evidence_summary, hyp_text, hypotheses, infer_parent, _ev, _short_obs
from .llm import call, parse_json
from .state import Hyp

K = config.K

PROMPT = """A scientist is investigating an unknown system. Below is the current state of their investigation - the \
hypotheses they have written so far and how their own experiments bear on each - followed by {k} candidate new \
hypotheses they could propose at this step, each with how the same experiments bear on it and how it relates to the \
existing hypotheses. All facts are computed from the scientist's own experiments.

Choose the ONE candidate that most opens a promising new direction while being compatible with the evidence so far: \
prefer a candidate that can account for what the existing hypotheses cannot, without contradicting what the evidence \
already supports. A restatement of an existing hypothesis, or a direction the experiments already contradict, is less \
valuable. Do not judge by which candidate you believe is the true law; judge by the evidence and the direction.

CURRENT STATE:
{state}

CANDIDATES:
{cands}

Return JSON only: {{"choice": <candidate number>, "why": "<one sentence>"}}"""


def _candidate_facts(st, d, artifact, cid):
    norm = C.normalise(artifact)
    try:
        key = d.key(norm)
    except Exception:  # noqa: BLE001
        key = ("unparsed", hash(norm))
    h = Hyp(cid, 0, "candidate", norm, key, 0, [], [])
    lines = []
    same = next((g for g in st.hyps if g.key == key), None)
    if same is not None:
        lines.append(f"relation: the same as existing {same.id}")
    else:
        p = infer_parent(st, h) if st.hyps else None
        if p is None:
            lines.append("relation: a new direction (not a revision of any existing hypothesis)")
        else:
            parent = next(g for g in st.hyps if g.id == p)
            lines.append(f"relation: a revision of existing {p}")
            lines += ["  " + x for x in edge_lines(st, d, parent, h)]
    if st.obs:
        lines.append("evidence: " + evidence_summary(st, h))
        bad = [o for o, r, _ in _ev(st, h) if r == "conflict"]
        if bad:
            lines.append("  conflicts with: " + ", ".join(_short_obs(d, o) for o in bad[:6]) + (" ..." if len(bad) > 6 else ""))
        # experiments that no existing hypothesis explains but this candidate does
        if st.hyps:
            lab = {g.id: {o.id: r for o, r, _ in _ev(st, g)} for g in st.hyps}
            unexplained = [o for o in st.obs if all(lab[g][o.id] != "support" for g in lab)]
            newly = [o for o, r, _ in _ev(st, h) if r == "support" and o in unexplained]
            if newly:
                lines.append("  supported by experiments no existing hypothesis explains: "
                             + ", ".join(_short_obs(d, o) for o in newly[:6]))
    else:
        lines.append("evidence: no experiments yet")
    lines.append("content:")
    lines += hyp_text(d, h)
    return lines



def choose(st, d, artifacts, model, effort=None):
    """(index of the selected candidate, record of the choice); index 0 if the call fails."""
    state = "\n".join(hypotheses(st, d)) if st.hyps else "(no hypotheses yet)"
    if st.obs:
        state = f"{len(st.obs)} experiments so far.\n" + state
    blocks = []
    for i, a in enumerate(artifacts, 1):
        try:
            f = _candidate_facts(st, d, a, f"C{i}")
        except Exception as e:  # noqa: BLE001
            f = [f"(facts unavailable: {type(e).__name__})", "content:"] + ["    " + x for x in code_core(a)]
        blocks.append(f"Candidate {i}:\n" + "\n".join(f))
    prompt = PROMPT.format(k=len(artifacts), state=state, cands="\n\n".join(blocks))
    rec = {"n": len(artifacts), "facts": [b[:1500] for b in blocks]}
    txt, usage = call(model, prompt, effort)
    out = parse_json(txt)
    try:
        i = int(out.get("choice")) - 1
        if not 0 <= i < len(artifacts):
            raise ValueError("choice out of range")
    except Exception as e:  # noqa: BLE001
        rec.update(choice=0, error=f"{type(e).__name__}: {str(e)[:200]}")
        return 0, rec
    rec.update(choice=i, why=str(out.get("why", ""))[:300], tokens=int(usage.get("total_tokens") or 0))
    return i, rec


# -- Hypothesis Search baseline -----------------------------------------------------------------------------------------
SELF_PROMPT = """You are helping a scientist who is investigating an unknown system. Below is the scientist's \
conversation so far, followed by {k} alternative next replies the scientist could make. Each reply proposes or \
revises a hypothesis.

Choose the ONE reply that you find most promising for eventually discovering the correct law of this system.

CONVERSATION SO FAR:
{context}

CANDIDATE REPLIES:
{cands}

Return JSON only: {{"choice": <number of the reply, 1-{k}>, "why": "<one sentence>"}}"""
CTX_CHARS, CAND_CHARS = 120000, 12000


def _transcript(context):
    """[(role, text)] -> text, keeping the beginning (task) and the most recent part within CTX_CHARS."""
    parts = [f"[{r}]\n{t}" for r, t in context if t]
    txt = "\n\n".join(parts)
    if len(txt) <= CTX_CHARS:
        return txt
    head = parts[0][: CTX_CHARS // 4] if parts else ""
    return head + "\n\n[... earlier turns omitted ...]\n\n" + txt[-(CTX_CHARS - len(head) - 50):]


def self_choose(context, replies, model, effort=None):
    """Hypothesis Search: the backbone picks the most promising of several sampled replies, from the conversation and
    the replies alone."""
    blocks = [f"Reply {i}:\n{(r or '')[:CAND_CHARS]}" for i, r in enumerate(replies, 1)]
    prompt = SELF_PROMPT.format(k=len(replies), context=_transcript(context or []), cands="\n\n".join(blocks))
    rec = {"n": len(replies), "mode": "hypothesis_search"}
    err, tok = None, 0
    for attempt in range(3):                                   # an unparsable answer is asked again
        txt, usage = call(model, prompt, effort)
        tok += int((usage or {}).get("total_tokens") or 0)
        out = parse_json(txt)
        try:
            i = int(out.get("choice")) - 1
            if not 0 <= i < len(replies):
                raise ValueError("choice out of range")
            rec.update(choice=i, why=str(out.get("why", ""))[:300], tokens=tok, attempts=attempt + 1)
            return i, rec
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"
    rec.update(choice=0, error=err, tokens=tok, attempts=3)
    return 0, rec
