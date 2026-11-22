"""Settings. Every default is the configuration used in the paper; environment variables override them.

  EPIC_METHOD       epicompass | react (no component) | hsearch (Hypothesis Search) | popper (Popper)       epicompass
  EPIC_K            candidates sampled at a hypothesis step for epistemic-gain selection (1 disables selection)   4
  EPIC_EFFORT       reasoning effort of the component's own LLM calls (the same backbone as the agent)         low
  EPIC_MAX_NOTES    routine epistemic feedback presentations per episode                                       8
  EPIC_SHOW         what routine feedback shows: full | tree (revision tree only) | none                      full
  EPIC_REVIEW       epistemically grounded self-verification at the first submission (1 / 0)                    1
  EPIC_EXPLORE      exploration state: "Open directions" in the feedback (1 / 0)                               1
  EPIC_BELIEF       belief update: llm (stated by the backbone) | formula (likelihood-based, no LLM call)      llm

LLM endpoint (any OpenAI-compatible chat-completions API, e.g. Gemini, OpenAI, or a local vLLM server):
  EPIC_BASE_URL     default: Gemini's OpenAI-compatible endpoint
  EPIC_API_KEY      or EPIC_API_KEYS (comma-separated, rotated); falls back to GEMINI_API_KEY / OPENAI_API_KEY
  EPIC_SYSTEM       optional system message prepended to every call (e.g. "Reasoning strength: low" for some models)
"""
import os


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


METHOD = os.environ.get("EPIC_METHOD", "epicompass")
K = _int("EPIC_K", 4)
EFFORT = os.environ.get("EPIC_EFFORT", "low")
MAX_NOTES = _int("EPIC_MAX_NOTES", 8)
SHOW = os.environ.get("EPIC_SHOW", "full")
REVIEW = os.environ.get("EPIC_REVIEW", "1") == "1"
EXPLORE = os.environ.get("EPIC_EXPLORE", "1") == "1"
BELIEF = os.environ.get("EPIC_BELIEF", "llm")

BASE_URL = os.environ.get("EPIC_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
SYSTEM = os.environ.get("EPIC_SYSTEM", "")


def api_keys():
    keys = os.environ.get("EPIC_API_KEYS") or os.environ.get("EPIC_API_KEY") or os.environ.get("GEMINI_API_KEY") \
        or os.environ.get("OPENAI_API_KEY") or ""
    return [k.strip() for k in keys.split(",") if k.strip()] or ["EMPTY"]
