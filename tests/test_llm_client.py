"""The Claude client drops settings a model rejects, and never needs the network in tests."""
from types import SimpleNamespace as NS

from credit_risk.genai.llm import AnthropicClient


class _Messages:
    def __init__(self, reject=()):
        self.reject, self.seen = reject, []

    def create(self, **kw):
        self.seen.append(set(kw))
        for k in self.reject:
            if k in kw:
                raise ValueError(f"{k}: not supported for this model")
        return NS(content=[NS(type="text", text="hello")], stop_reason="end_turn")


def _client(reject=()):
    c = AnthropicClient.__new__(AnthropicClient)
    c.model, c.client = "m", NS(messages=_Messages(reject))
    return c


def test_sends_temperature_and_no_thinking_by_default():
    c = _client()
    assert c.complete("s", "u", 50) == "hello"
    assert {"temperature", "thinking"} <= c.client.messages.seen[0] and c.last_stop_reason == "end_turn"


def test_drops_each_setting_the_model_rejects():
    c = _client(reject=("temperature", "thinking"))
    assert c.complete("s", "u", 50) == "hello"
    assert not ({"temperature", "thinking"} & c.client.messages.seen[-1])
