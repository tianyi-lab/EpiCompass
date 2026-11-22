"""Chat calls for the component's own steps (belief update, hypothesis selection, feedback writing, baselines).

Plain HTTP to an OpenAI-compatible chat-completions endpoint (no SDK, so the component runs inside each benchmark's own
environment). Every call uses the agent's own backbone and reasoning effort, so no capability is borrowed.
"""
from __future__ import annotations

import itertools
import json
import random
import re
import time
import urllib.error
import urllib.request

from . import config

_keys = itertools.cycle(config.api_keys())


def chat(model, messages, effort=None, json_mode=False, tries=8, timeout=300, **extra):
    """(text, usage) of one chat completion; retries on rate limits, server errors and timeouts."""
    if config.SYSTEM and not (messages and messages[0].get("role") == "system"):
        messages = [{"role": "system", "content": config.SYSTEM}] + list(messages)
    body = {"model": model, "messages": messages, **extra}
    if effort:
        body["reasoning_effort"] = effort
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    usage = {}
    for k in range(tries):
        req = urllib.request.Request(config.BASE_URL + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {next(_keys)}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            usage = out.get("usage") or {}
            txt = (((out.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
            if txt:
                return txt, usage
        except urllib.error.HTTPError as e:
            if e.code not in (408, 429, 500, 502, 503, 504):
                return "", {"error": f"HTTP {e.code}"}
        except Exception:  # noqa: BLE001 -- timeouts and connection resets are retried
            pass
        time.sleep(min(120, 5 * 2 ** k) * (0.5 + random.random()))
    return "", {"error": "no response"}


def call(model, prompt, effort=None):
    """One JSON-mode question to the backbone: (text, usage)."""
    return chat(model, [{"role": "user", "content": prompt}], effort or config.EFFORT, json_mode=True)


def parse_json(txt):
    """The first JSON object in a reply, or None."""
    m = re.search(r"\{.*\}", txt or "", re.S)
    try:
        return json.loads(m.group(0) if m else txt)
    except Exception:  # noqa: BLE001
        return None
