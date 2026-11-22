"""Wording of the findings (with the agent's own records and numbers; no verdicts, no suggested answer) and the
entry point that turns a set of findings into the feedback text returned to the agent."""
from __future__ import annotations

HEADER = ("[epistemic notes - computed only from your own experiments, your own models/laws and your own statements; "
          "they do not say which hypothesis is right]")


def _fmt(x):
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "non-finite"
    return f"{x:.3g}"


def _odds(r):
    if r >= 1e4:
        return ">10^4:1"
    return f"{r:.3g}:1" if r >= 1 else f"1:{1 / max(r, 1e-300):.3g}"


def _val(x):
    if isinstance(x, (list, tuple)):
        return "(" + ", ".join(_fmt(float(y)) for y in x) + ")"
    return _fmt(float(x))


def one(st, v, d):
    """One finding as a bullet line ("" if it has no wording of its own; S0, S1 and M1 are shown per hypothesis)."""
    x = v.detail
    if v.kind == "S3":
        return (f"- You went back to the form of {v.subject}. When you last wrote it, on the experiments you had then, "
                f"its mean error was {_fmt(x['total'])} vs {_fmt(x['other_total'])} for your {x['other']}, and no "
                f"experiment has been run since.")
    if v.kind == "D1":
        return (f"- Your latest experiments changed which of your hypotheses the evidence favours: {x['new']} over "
                f"{x['old']}, odds {_odds(x['odds_before'])} before them, {_odds(x['odds_after'])} after them.")
    if v.kind == "D2":
        return (f"- Your latest experiments moved the odds {x['factor']:.3g}x toward {x['leader']} and away from the Main "
                f"hypothesis {x['cur']}; they are now {_odds(x['odds_after'])} for {x['leader']}.")
    if v.kind == "D3":
        o = next(y for y in st.obs if y.id == v.subject)
        more = f" (and {len(x['also'])} more such separations this round)" if x.get("also") else ""
        return (f"- Your latest {d.describe_obs(o)}{more} separates two hypotheses your earlier experiments could not: "
                f"error {_fmt(x['e_win'])} for {x['win']} vs {_fmt(x['e_lose'])} for {x['lose']}.")
    if v.kind == "D4":
        o = next(y for y in st.obs if y.id == v.subject)
        es = ", ".join(f"{k} {_fmt(e)}" for k, e in sorted(x["errs"].items()))
        more = f" (and {len(x['also'])} more of this round's experiments)" if x.get("also") else ""
        return (f"- None of your hypotheses explains your latest {d.describe_obs(o)}{more}: each misses it by far more "
                f"than your earlier experiments (errors: {es}).")
    if v.kind == "R1":
        s = (f"- {v.subject} and {x['rival']} fit your experiments about equally (mean error {_fmt(x['total'])} vs "
             f"{_fmt(x['rival_total'])}): your data do not yet separate them.")
        if x.get("disc"):
            s += " Experiments where they disagree most: " + "; ".join(
                f"{q['input_text']} -> {q['what']}: {_val(q['pred_a'])} under {v.subject} vs {_val(q['pred_b'])} under "
                f"{x['rival']}" for q in x["disc"]) + "."
        return s
    return ""


def footer(extra=""):
    return ("Before this submission is scored you may revise it, investigate further if you have turns left, "
            "or submit it unchanged. Your next submission is final." + (" " + extra if extra else ""))


def render(st, findings, d, at_submit=False, extra=""):
    """The feedback for these findings: the epistemic-state card (card.py), plus the verification prompt at the first
    submission. "" when there is nothing to show."""
    if not findings:
        return ""
    from . import card
    w = getattr(st, "card_writer", None)
    body = w(st, findings, d, at_submit) if w is not None else card.card_a(st, findings, d, at_submit)
    if not body.strip():
        return ""
    return body + ("\n\n" + footer(extra) if at_submit else "")
