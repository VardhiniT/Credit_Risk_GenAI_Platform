"""Thin LLM wrapper. The assistant works without an API key (rule-based routing and template answers).
With ANTHROPIC_API_KEY set, the same code uses Claude for intent detection and for writing explanations.
"""
import os

DEFAULT_MODEL = "claude-haiku-5-5"


class AnthropicClient:
    name = "anthropic"
    last_stop_reason = None

    def __init__(self, model=None, api_key=None):
        import anthropic                                   # imported only when actually used
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.model = model or os.environ.get("CREDIT_LLM_MODEL", DEFAULT_MODEL)

    def complete(self, system, user, max_tokens=400):
        kw = dict(model=self.model, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": user}],
                  temperature=0, thinking={"type": "disabled"})     # short, deterministic tasks: no hidden reasoning tokens
        for _ in range(3):                                          # drop any setting this model does not accept
            try:
                r = self.client.messages.create(**kw)
                break
            except Exception as e:
                msg = str(e).lower()
                dropped = [k for k in ("temperature", "thinking") if k in kw and k in msg]
                if not dropped:
                    raise
                for k in dropped:
                    kw.pop(k)
        else:
            raise RuntimeError("the model rejected the request settings")
        self.last_stop_reason = getattr(r, "stop_reason", None)
        return "".join(getattr(b, "text", "") or "" for b in r.content if getattr(b, "type", "text") == "text").strip()


def get_client():
    """Return an LLM client if a key and the SDK are available, else None (fallback mode)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        return AnthropicClient()
    except Exception:
        return None


def check():
    """Call Claude once and report what happened. Used by:  python -m credit_risk.genai.chat --check-llm"""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY is not set in this terminal window (run the export command again)."
    try:
        client = AnthropicClient()
    except Exception as e:
        return False, f"Could not start the client: {type(e).__name__}: {e}"
    try:
        out = client.complete("Reply with exactly one word.", "hello", 20)
        return True, f"OK. Model {client.model} answered: {out!r}"
    except Exception as e:
        return False, f"Call failed with model {client.model}: {type(e).__name__}: {str(e)[:300]}"
