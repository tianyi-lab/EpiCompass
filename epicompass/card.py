"""The epistemic feedback s_t = RENDER(T_t): a fixed-schema card summarising the epistemic tree.

Sections (in this order; an empty one is left out):
  Hypotheses        the leading hypotheses as a revision tree: Main / Alternative, belief, how many experiments support
                    or conflict with each, the conflicting experiments (observed vs predicted), and for a revision what
                    it changed and which experiments it fixes or breaks
  Conflicts         experiments no hypothesis explains; a return to a hypothesis already shown worse
  Still undecided   rival hypotheses the experiments do not separate yet
  Changed           what the latest experiments changed
The facts are computed by code. With a card writer (the default), the backbone may shorten and order the facts of the
non-hypothesis sections but may only restate them; the Hypotheses section is always kept verbatim. On any failure the
code-only card is used.
"""
from __future__ import annotations

import difflib
import json
import re

from . import update as T
from .llm import call, parse_json

MAX_HYPS = 5                 # hypotheses per routine card
MAX_CONFLICTS = 2            # conflicting experiments listed per hypothesis
MAX_SHARED = 4               # experiments listed in "every hypothesis below conflicts with ..."
PARENT_MIN_SIM = 0.35        # below this similarity a new hypothesis starts a new branch of the tree
HYP_LINES, HYP_CHARS = 14, 900

SECTION = {"S3": "Conflicts", "D4": "Conflicts", "R1": "Still undecided", "D3": "Still undecided", "D2": "Changed"}
ORDER = ["Hypotheses", "Conflicts", "Still undecided", "Changed"]
HEAD = ("[Epistemic state - computed only from your own experiments, your own models/laws and your own statements; "
        "it does not say which hypothesis is right]")


def _fmt(x):
    from .render import _fmt as f
    return f(x)


# -- hypothesis content -------------------------------------------------------------------------------------------------
# only lines that carry no logic are dropped: comments, imports, the def header, docstrings, and the integrator plumbing
# of a trajectory law (state vector, solve_ivp, unpacking). Control flow is kept.
_BOILER = re.compile(r"^\s*(#|import |from \S+ import |\"\"\"|\'\'\'|y0\s*=|sol\s*=|.*solve_ivp\(|"
                     r"return \[y\[|return sol\.|r_vec\s*=\s*y\[)")


def _drop_docstrings(src):
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            b = node.body[0]
            if isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant) and isinstance(b.value.value, str):
                drop.update(range(b.lineno, (b.end_lineno or b.lineno) + 1))
    return "\n".join(l for i, l in enumerate(src.splitlines(), 1) if i not in drop)


def code_core(src, max_lines=HYP_LINES, max_chars=HYP_CHARS):
    """The lines of a law that carry its logic, verbatim, with their relative indentation."""
    out, skip = [], False
    for ln in _drop_docstrings(src or "").splitlines():
        if re.match(r"^def fit_parameters", ln):
            skip = True
        elif re.match(r"^def ", ln):
            skip = False
        if skip or not ln.strip() or _BOILER.match(ln):
            continue
        ln = re.sub(r"\s+#.*$", "", ln).rstrip()
        if ln.strip():
            out.append(ln)
    ind = min((len(x) - len(x.lstrip()) for x in out), default=0)
    out = [x[ind:] for x in out]
    keep, n = [], 0
    for ln in out:
        if len(keep) >= max_lines or n + len(ln) > max_chars:
            keep.append(f"... ({len(out) - len(keep)} more lines)")
            break
        keep.append(ln)
        n += len(ln)
    return keep


def hyp_repr(d, artifact):
    """The lines that say what a hypothesis is, used for tree edges and change summaries."""
    try:
        return list(d.describe_hyp(artifact)) if hasattr(d, "describe_hyp") else code_core(artifact, 40, 4000)
    except Exception:  # noqa: BLE001
        return code_core(artifact, 40, 4000)


def hyp_text(d, h):
    try:
        lines = d.describe_hyp(h.artifact) if hasattr(d, "describe_hyp") else code_core(h.artifact)
    except Exception:  # noqa: BLE001
        lines = code_core(h.artifact)
    return [f"    {x}" for x in lines]


def infer_parent(st, h):
    """The existing hypothesis h most resembles in its core content (h revises it); None starts a new branch."""
    me = "\n".join(hyp_repr(st.d, h.artifact))
    best, sim = None, 0.0
    for g in st.hyps:
        r = difflib.SequenceMatcher(None, me, "\n".join(hyp_repr(st.d, g.artifact))).ratio()
        if r > sim or (r == sim and best is not None and g.last_turn > best.last_turn):
            best, sim = g, r
    return best.id if best is not None and sim >= PARENT_MIN_SIM else None


# -- evidence -----------------------------------------------------------------------------------------------------------
def _ev(st, h):
    from .checks import evidence
    k = ("ev", h.id, hash(h.artifact), len(st.obs), len(st.hyps))
    c = st.__dict__.setdefault("_ev_cache", {})
    if k not in c:
        c[k] = evidence(st, h)
    return c[k]


def evidence_summary(st, h):
    n = len(st.obs)
    if n == 0:
        return "no experiments yet: no evidence for or against it"
    ev = _ev(st, h)
    a = sum(r == "support" for _, r, _ in ev)
    b = sum(r == "conflict" for _, r, _ in ev)
    return f"of your {n} experiments, {a} support it, {b} conflict with it, {n - a - b} give no evidence either way"


def _short_obs(d, o):
    t = d.describe_obs(o).replace("experiment ", "", 1)
    if t.startswith(("iteration_", "round ")) and ":" in t:
        return t.split(":")[0][:40]
    return t[:80]


def conflict_lines(st, d, h, exclude=()):
    """Up to MAX_CONFLICTS of h's conflicting experiments (observed vs predicted). Experiments every shown hypothesis
    conflicts with (exclude) are listed once above the hypotheses."""
    allbad = sorted([(o, e) for o, r, e in _ev(st, h) if r == "conflict"], key=lambda x: -(x[1] if x[1] is not None else 0))
    bad = [x for x in allbad if x[0].id not in exclude] or allbad[:1]
    out = []
    for o, e in bad[:MAX_CONFLICTS]:
        try:
            cmp_ = d.compare(h.artifact, o)
        except Exception:  # noqa: BLE001
            cmp_ = ""
        if e is None or e == float("inf"):
            out.append(f"    conflict: {d.describe_obs(o)} (no finite prediction)")
        else:
            out.append(f"    conflict: {d.describe_obs(o)} (error {_fmt(e)}" + (f"; {cmp_}" if cmp_ else "") + ")")
    if len(bad) > MAX_CONFLICTS:
        out.append(f"    ... and {len(bad) - MAX_CONFLICTS} more conflicting experiments")
    return out


_KEY = re.compile(r"\b(acc\w*|force|F|a|dydt|rate|return)\b\s*[\[=(]|\*\*|->|<->|\breturn\b")


def _key_lines(lines, k=2):
    """The k changed lines that most likely carry the law, in their original order."""
    def score(x):
        if not x or x[0] in "`<#" or x.startswith(("...", "Here", "I ")):
            return -9
        return (3 if _KEY.search(x) else 0) + (1 if any(op in x for op in ("*", "/", "+", "-")) else 0) \
            - (2 if "params.get" in x else 0)
    ranked = sorted(range(len(lines)), key=lambda i: -score(lines[i]))
    keep = sorted(i for i in ranked[:k] if score(lines[i]) > -9)
    return [lines[i] for i in keep]


def edge_lines(st, d, parent, child):
    """A tree edge: what the revision parent -> child changed, and which experiments it fixes or breaks."""
    a, b = hyp_repr(d, parent.artifact), hyp_repr(d, child.artifact)
    removed = _key_lines([x.strip() for x in a if x not in b])
    added = _key_lines([x.strip() for x in b if x not in a])
    ch = []
    if removed:
        ch.append("removed: " + " | ".join(x[:90] for x in removed))
    if added:
        ch.append("added: " + " | ".join(x[:90] for x in added))
    out = ["change from " + parent.id + ": " + ("; ".join(ch) if ch else "constants or formatting only")]
    pa = {o.id: r for o, r, _ in _ev(st, parent)}
    fixed = [o for o, r, _ in _ev(st, child) if r == "support" and pa.get(o.id) == "conflict"]
    broke = [o for o, r, _ in _ev(st, child) if r == "conflict" and pa.get(o.id) == "support"]
    if fixed or broke:
        f = lambda xs: f"{len(xs)} (" + ", ".join(_short_obs(d, o) for o in xs[:3]) + (", ..." if len(xs) > 3 else "") + ")"
        out.append(("fixed " + f(fixed) if fixed else "fixed none") + "; " + ("broke " + f(broke) if broke else "broke none"))
    return out


# -- main hypothesis ----------------------------------------------------------------------------------------------------
def top_hyp(st):
    """The hypothesis with the highest belief, when a belief compares >= 2 hypotheses; else None."""
    ids = {h.id for h in st.hyps}
    b = {k: v for k, v in (st.cred or {}).items() if k in ids}
    return max(b, key=b.get) if len(b) >= 2 else None


def main_change_lines(st, d, findings):
    """The explicit statement that the main hypothesis changed (finding M1), with its content."""
    f = next((v for v in findings if v.kind == "M1"), None)
    if f is None:
        return []
    x = f.detail
    h = next((g for g in st.hyps if g.id == x["top"]), None)
    if h is None:
        return []
    prev = (f"; previously {x['prev']}, now {x['prev_b']:.2f}" if x.get("prev") else
            f"; your latest hypothesis {x['latest']} has {x['latest_b']:.2f}" if x.get("latest") else "")
    return [f"Your belief has changed: given your experiments, the most likely hypothesis is now {h.id} "
            f"(belief {x['b']:.2f}{prev}):"] + hyp_text(d, h)


# -- the Hypotheses section (the epistemic tree) -------------------------------------------------------------------------
def hypotheses(st, d, n=MAX_HYPS):
    """The leading hypotheses as a revision tree, Main first, then by belief."""
    rows = []
    for h in st.hyps:
        e = T.errors(st, h)
        rows.append((h, st.cred.get(h.id), None if e is None else T.total(e)))
    cur = st.main_id or top_hyp(st) or (st.current.id if st.current is not None else None)
    if cur not in {h.id for h in st.hyps}:
        cur = top_hyp(st) or (st.current.id if st.current is not None else None)
    rows.sort(key=lambda r: (r[0].id != cur, -(r[1] or 0.0), r[2] if r[2] is not None else float("inf")))
    out = []
    show_b = len(rows) >= 2
    common = set()
    if len(rows[:n]) >= 2 and st.obs:
        sets = [{o.id for o, r, _ in _ev(st, h) if r == "conflict"} for h, _, _ in rows[:n]]
        common = set.intersection(*sets)
        if common:
            names = [d.describe_obs(o) for o in st.obs if o.id in common]
            shown = "; ".join(names[:MAX_SHARED]) + (f"; and {len(names) - MAX_SHARED} more" if len(names) > MAX_SHARED else "")
            out.append(f"- Every hypothesis below conflicts with {len(names)} of your experiments: {shown}")
    shown = rows[:n]
    depth, edge = {}, {}
    byid = {h.id: h for h in st.hyps}
    if len(shown) >= 2:
        ids = {h.id for h, _, _ in shown}

        def anc(h):                                     # nearest shown ancestor
            p = h.parent
            while p is not None and p not in ids:
                p = byid[p].parent if p in byid else None
            return p
        kids = {}
        for r in shown:
            kids.setdefault(anc(r[0]), []).append(r)
        order = []

        def walk(pid, dep):
            for r in sorted(kids.get(pid, []), key=lambda r: (r[0].id != cur, -(r[1] or 0.0))):
                depth[r[0].id], edge[r[0].id] = dep, pid
                order.append(r)
                walk(r[0].id, dep + 1)
        walk(None, 0)
        if len(order) == len(shown):
            shown = order
        out.insert(1 if common else 0, "- (tree: each child is a revision of its parent)")
    for h, b, tot in shown:
        ind = "    " * depth.get(h.id, 0)
        pre = "└─ " if depth.get(h.id, 0) else "- "
        role = "[Main]" if h.id == cur else "[Alternative]"
        bel = f"belief {'n/a' if b is None else f'{b:.2f}'} | " if show_b else ""
        block = [f"{pre}{h.id} {role}: {bel}" + evidence_summary(st, h)]
        if edge.get(h.id):
            block += ["    " + x for x in edge_lines(st, d, byid[edge[h.id]], h)]
        block += hyp_text(d, h) + conflict_lines(st, d, h, exclude=common)
        out += [ind + block[0] if ind else block[0]] + [ind + x for x in block[1:]]
    if len(rows) > n:
        out.append(f"- ({len(rows) - n} more, with lower belief)")
    return out


def facts(st, findings, d):
    """{section: [lines]} from the findings (S0 / S1 / M1 appear through the Hypotheses section instead)."""
    from .render import one
    sec = {s: [] for s in ORDER}
    for v in findings:
        if v.kind in ("S0", "S1", "M1", "D1"):
            continue
        t = one(st, v, d)
        if t:
            sec[SECTION.get(v.kind, "Conflicts")].append(t.strip())
    return sec


def _with_hypotheses(st, d, findings, at_submit, sec):
    if at_submit or any(sec[s] for s in ORDER[1:]) or any(v.kind in ("S0", "S1", "M1") for v in findings):
        sec["Hypotheses"] = hypotheses(st, d)
    return sec


def _head(st, d, findings):
    unit = f" Errors are {d.error_unit}." if getattr(d, "error_unit", "") else ""
    mc = main_change_lines(st, d, findings)
    return ["", HEAD + unit] + (["", *mc] if mc else [])


def card_a(st, findings, d, at_submit=False):
    """The card filled by code (deterministic, facts only)."""
    sec = _with_hypotheses(st, d, findings, at_submit, facts(st, findings, d))
    lines = _head(st, d, findings)
    for s in ORDER:
        if sec[s]:
            lines += ["", f"## {s}"] + sec[s]
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


PROMPT = """You maintain a scientist's epistemic-state card. Below are FACTS computed from the scientist's own \
experiments, hypotheses and statements, grouped by section. Write the card as JSON with exactly these keys: \
"Hypotheses", "Conflicts", "Your claims", "Still undecided", "Changed" - each a list of short bullet strings.

Rules: restate only what the FACTS say, keeping every hypothesis id, number, experiment input and quote that you use \
exactly as given; you may merge, shorten and order the items by importance; leave a key's list empty if it has no facts. \
Do NOT say which hypothesis is right, do NOT suggest a law, a form, a parameter value or an experiment that is not in \
the FACTS, and add no advice.

FACTS:
{facts}

Return JSON only."""


class CardWriter:
    """The backbone writes the non-hypothesis sections from the same facts (restating them only)."""

    def __init__(self, model, effort=None):
        self.model, self.effort = model, effort
        self.calls, self.tokens, self.errors = 0, 0, 0

    def __call__(self, st, findings, d, at_submit=False):
        sec = _with_hypotheses(st, d, findings, at_submit, facts(st, findings, d))
        given = {s: v for s, v in sec.items() if v and s != "Hypotheses"}
        if not given:
            return card_a(st, findings, d, at_submit) if sec.get("Hypotheses") else ""
        txt, usage = call(self.model, PROMPT.format(facts=json.dumps(given, indent=1, ensure_ascii=False)), self.effort)
        self.calls += 1
        self.tokens += int(usage.get("total_tokens") or 0)
        out = parse_json(txt)
        if not isinstance(out, dict) or not any(isinstance(out.get(s), list) and out.get(s) for s in ORDER):
            self.errors += 1
            return card_a(st, findings, d, at_submit)
        lines = _head(st, d, findings)
        for s in ORDER:
            if s == "Hypotheses":
                if sec.get("Hypotheses"):
                    lines += ["", "## Hypotheses"] + sec["Hypotheses"]
                continue
            items = [str(x).strip() for x in (out.get(s) or []) if str(x).strip()]
            if items:
                lines += ["", f"## {s}"] + [x if x.startswith("- ") else f"- {x}" for x in items]
        return "\n".join(lines)
