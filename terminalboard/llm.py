"""Optional LLM assistant — natural-language navigation + analysis.

Uses LiteLLM as a single interface over every provider (OpenAI / Anthropic /
Gemini / OpenRouter / Groq / local Ollama / …): the *model string* selects the
provider and the *api_key* is passed alongside, so there is no provider-specific
code here. LiteLLM is an optional dependency (``terminalboard[llm]``) and is
imported lazily, so the base install is unaffected.

The module is deliberately UI-free and network-injectable: every call takes an
optional ``complete`` callable (defaulting to ``litellm.completion``) so tests
can run with a fake and no network.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# Privacy: litellm's only non-provider network call is fetching a public model
# pricing JSON from raw.githubusercontent.com at import. Use its bundled local
# snapshot instead, so the ONLY traffic is to the provider the user configured.
# (Affects nothing but the $-estimate, which may lag provider price changes.
# setdefault → a user who exports it =false gets the live fetch back.)
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")


# --- availability + config persistence --------------------------------------

def is_available() -> bool:
    """True if LiteLLM is importable (the ``[llm]`` extra is installed)."""
    try:
        import litellm  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class LLMConfig:
    model: str = ""
    api_key: str = ""
    api_base: str = ""

    def ok(self) -> bool:
        return bool(self.model.strip())


def config_path() -> str:
    base = (os.environ.get("XDG_STATE_HOME")
            or os.path.expanduser("~/.local/state"))
    return os.path.join(base, "terminalboard", "llm.json")


def load_config() -> Optional[LLMConfig]:
    try:
        with open(config_path()) as f:
            d = json.load(f)
        cfg = LLMConfig(model=d.get("model", ""), api_key=d.get("api_key", ""),
                        api_base=d.get("api_base", ""))
        return cfg if cfg.ok() else None
    except Exception:
        return None


def save_config(cfg: LLMConfig) -> None:
    path = config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"model": cfg.model, "api_key": cfg.api_key,
                   "api_base": cfg.api_base}, f, indent=2)
    try:
        os.chmod(path, 0o600)        # keep the API key out of other users' reach
    except OSError:
        pass


# --- the action schema (single source for tools + executor) -----------------
#
# Each spec is an OpenAI-style function tool. The executor that applies them
# lives in app.py (it needs the live App); names/params must stay in sync.

ACTION_SPECS = [
    {"name": "set_tag_filter",
     "description": "Filter which metric tags are shown. Pattern supports "
                    "substring, globs (train/*loss*), OR (a|b), AND (space), "
                    "NOT (!x), and /regex/. Pass null to clear.",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": ["string", "null"]}}, "required": ["pattern"]}},
    {"name": "set_experiment_filter",
     "description": "Filter which experiments/runs are shown (same pattern "
                    "grammar as tags). Pass null to clear.",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": ["string", "null"]}}, "required": ["pattern"]}},
    {"name": "set_type",
     "description": "Show only one data type, or all.",
     "parameters": {"type": "object", "properties": {"kind": {"type": "string",
         "enum": ["all", "scalar", "histogram", "text", "pr_curve"]}},
         "required": ["kind"]}},
    {"name": "set_smoothing",
     "description": "Set scalar EMA smoothing weight in [0, 0.99] (0 = off).",
     "parameters": {"type": "object", "properties": {
         "value": {"type": "number"}}, "required": ["value"]}},
    {"name": "set_zoom",
     "description": "Set how many panels per page (snaps to 1,2,4,6,9,12,16,24,36).",
     "parameters": {"type": "object", "properties": {
         "panels": {"type": "integer"}}, "required": ["panels"]}},
    {"name": "set_xaxis",
     "description": "Scalar x-axis: by step or by wall-clock time.",
     "parameters": {"type": "object", "properties": {"axis": {"type": "string",
         "enum": ["step", "time"]}}, "required": ["axis"]}},
    {"name": "set_logy",
     "description": "Toggle log-scale Y for scalar panels.",
     "parameters": {"type": "object", "properties": {
         "on": {"type": "boolean"}}, "required": ["on"]}},
    {"name": "set_distribution",
     "description": "Show histograms as distribution bands (true) or heatmap (false).",
     "parameters": {"type": "object", "properties": {
         "on": {"type": "boolean"}}, "required": ["on"]}},
    {"name": "open_detail",
     "description": "Open a single tag full-screen (must be an existing tag).",
     "parameters": {"type": "object", "properties": {
         "tag": {"type": "string"}}, "required": ["tag"]}},
    {"name": "close_detail",
     "description": "Return from the full-screen detail view to the grid.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "open_hparams",
     "description": "Open the HParams table (runs × hyperparameters × metrics).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "goto_page",
     "description": "Jump to a 1-based page of the tag grid.",
     "parameters": {"type": "object", "properties": {
         "page": {"type": "integer"}}, "required": ["page"]}},
]

ACTION_NAMES = {s["name"] for s in ACTION_SPECS}


def build_tools() -> List[dict]:
    return [{"type": "function", "function": s} for s in ACTION_SPECS]


# Curated, current, light-first picks (these post-date litellm's frozen price map,
# so we list them explicitly; they float to the top of the picker).
CURATED_MODELS = [
    ("gpt-5.4-nano", "OpenAI"),
    ("gpt-5.4-mini", "OpenAI"),
    ("gpt-4o-mini", "OpenAI"),
    ("anthropic/claude-haiku-4-5", "Anthropic"),
    ("anthropic/claude-sonnet-4-6", "Anthropic"),
    ("gemini/gemini-3.5-flash", "Google"),
    ("gemini/gemini-3.1-flash-lite", "Google"),
    ("deepseek/deepseek-v4-flash", "DeepSeek"),
    ("deepseek/deepseek-v4-pro", "DeepSeek"),
    ("openrouter/qwen/qwen3.6-35b-a3b", "OpenRouter"),
    ("ollama/llama3", "Ollama (local)"),
    ("hosted_vllm/Qwen/Qwen3.6-27B", "vLLM (self-host)"),
]

_CATALOG = None


def model_catalog():
    """(model, provider) pairs for the picker: curated picks first, then every
    chat model litellm knows about (for search). Cached."""
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    items = list(CURATED_MODELS)
    seen = {m for m, _ in items}
    try:
        import litellm
        for name, meta in sorted(litellm.model_cost.items()):
            if meta.get("mode") != "chat" or name in seen:
                continue
            items.append((name, meta.get("litellm_provider", "")))
            seen.add(name)
    except Exception:
        pass
    _CATALOG = items
    return items


SYSTEM_PROMPT = (
    "You are an assistant embedded in terminalboard, a terminal TensorBoard "
    "viewer. You help the user navigate the dashboard and analyze their "
    "experiment results.\n"
    "- To change what is shown (filter tags/experiments, pick a data type, "
    "smooth, zoom, open a tag, open the HParams table, etc.), CALL THE TOOLS. "
    "You may call several in one turn.\n"
    "- To answer a question or give analysis/suggestions, reply with concise "
    "text. You can do both (navigate AND explain) in the same turn.\n"
    "- Use the exact tag and experiment names from the context. Prefer precise "
    "filters. Keep text answers short and specific; use plain text.\n"
)


def build_messages(context: str, question: str, history=None) -> List[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT
             + "\n\nCurrent dashboard context:\n" + context}]
    if history:
        msgs += list(history)
    msgs.append({"role": "user", "content": question})
    return msgs


# --- response helpers (tolerate dict- or object-style responses) ------------

def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _default_complete(**kwargs):
    import litellm
    # Keep litellm from printing debug/provider banners to stdout — in the
    # alternate-screen TUI any stray print corrupts the display. (We can't use
    # redirect_stdout here: the call streams from a worker thread while the main
    # thread draws to the same stdout.)
    litellm.suppress_debug_info = True
    try:
        litellm.set_verbose = False
    except Exception:
        pass
    return litellm.completion(**kwargs)


def _call(config: LLMConfig, complete, **kw):
    """Common kwargs for a LiteLLM call (omit empty key/base)."""
    if config.api_key:
        kw["api_key"] = config.api_key
    if config.api_base:
        kw["api_base"] = config.api_base
    return complete(model=config.model, **kw)


def validate(config: LLMConfig, *, complete: Optional[Callable] = None
             ) -> Tuple[bool, str]:
    """A tiny ping to check the model/key work. Returns (ok, error)."""
    complete = complete or _default_complete
    try:
        _call(config, complete,
              messages=[{"role": "user", "content": "ping"}], max_tokens=5)
        return True, ""
    except Exception as e:                       # pragma: no cover - network
        return False, str(e)


def ask(config: LLMConfig, messages: List[dict], tools: List[dict], *,
        complete: Optional[Callable] = None, temperature: float = 0.2,
        max_tokens: int = 1024) -> dict:
    """Run one completion. Returns {text, tool_calls:[(name,args)], usage, message}."""
    complete = complete or _default_complete
    resp = _call(config, complete, messages=messages, tools=tools,
                 tool_choice="auto", temperature=temperature,
                 max_tokens=max_tokens)
    choice = _get(resp, "choices")[0]
    msg = _get(choice, "message")
    text = _get(msg, "content") or ""
    tool_calls = []
    for tc in (_get(msg, "tool_calls") or []):
        fn = _get(tc, "function")
        name = _get(fn, "name")
        raw = _get(fn, "arguments")
        if isinstance(raw, str):
            try:
                args = json.loads(raw) if raw.strip() else {}
            except Exception:
                args = {}
        else:
            args = raw or {}
        if name in ACTION_NAMES:
            tool_calls.append((name, args))
    usage = _get(resp, "usage")
    return {"text": text.strip(), "tool_calls": tool_calls, "usage": usage,
            "message": msg}


def ask_stream(config: LLMConfig, messages: List[dict], tools: List[dict], *,
               complete: Optional[Callable] = None, on_delta: Optional[Callable] = None,
               temperature: float = 0.2, max_tokens: int = 1024) -> dict:
    """Like ``ask`` but streams: calls ``on_delta(text_fragment)`` as content
    arrives, and reassembles tool calls from deltas. Returns the same dict."""
    complete = complete or _default_complete
    stream = _call(config, complete, messages=messages, tools=tools,
                   tool_choice="auto", temperature=temperature,
                   max_tokens=max_tokens, stream=True,
                   stream_options={"include_usage": True})
    parts: List[str] = []
    frags: dict = {}            # index -> {"name", "args"}
    usage = None
    for chunk in stream:
        choices = _get(chunk, "choices") or []
        if choices:
            delta = _get(choices[0], "delta")
            if delta is not None:
                c = _get(delta, "content")
                if c:
                    parts.append(c)
                    if on_delta:
                        on_delta(c)
                for tc in (_get(delta, "tool_calls") or []):
                    idx = _get(tc, "index") or 0
                    fn = _get(tc, "function")
                    slot = frags.setdefault(idx, {"name": None, "args": ""})
                    nm = _get(fn, "name") if fn else None
                    if nm:
                        slot["name"] = nm
                    ar = _get(fn, "arguments") if fn else None
                    if ar:
                        slot["args"] += ar
        u = _get(chunk, "usage")
        if u:
            usage = u
    tool_calls = []
    for idx in sorted(frags):
        slot = frags[idx]
        if slot["name"] not in ACTION_NAMES:
            continue
        try:
            args = json.loads(slot["args"]) if slot["args"].strip() else {}
        except Exception:
            args = {}
        tool_calls.append((slot["name"], args))
    return {"text": "".join(parts).strip(), "tool_calls": tool_calls,
            "usage": usage, "cost": estimate_cost(config.model, usage)}


def estimate_cost(model: str, usage) -> Optional[float]:
    if not usage:
        return None
    try:
        import contextlib
        import io
        import litellm
        pt = _get(usage, "prompt_tokens") or 0
        ct = _get(usage, "completion_tokens") or 0
        # litellm prints provider hints to stdout/stderr for unknown models —
        # silence it so it can't corrupt the alternate-screen TUI.
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            pc, cc = litellm.cost_per_token(model=model, prompt_tokens=pt,
                                            completion_tokens=ct)
        return (pc or 0.0) + (cc or 0.0)
    except Exception:
        return None


def friendly_error(exc) -> str:
    """Map a provider/LiteLLM exception to a short, actionable message."""
    s = str(exc)
    low = s.lower()
    if any(k in low for k in ("api key", "authentication", "401", "unauthorized")):
        return "Auth failed — check your API key (press A to reconfigure)."
    if "rate limit" in low or "429" in low:
        return "Rate limited — wait a moment and try again."
    if any(k in low for k in ("connection", "timeout", "timed out", "network")):
        return "Network error — check your connection / api_base."
    if "not found" in low or "404" in low or "does not exist" in low:
        return "Model not found — check the model string (A to reconfigure)."
    return s


def usage_summary(usage) -> str:
    if not usage:
        return ""
    pt = _get(usage, "prompt_tokens") or 0
    ct = _get(usage, "completion_tokens") or 0
    if not (pt or ct):
        return ""
    return f"{pt}+{ct} tok"
