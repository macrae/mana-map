"""The model, behind a seam thin enough that CI never needs a key.

`Turn` is the whole contract: given a system prompt, a message list and a tool
block, yield events. Two implementations — one that calls Anthropic, one that
replays a script. Everything above this file is written against the protocol, so
the loop, the tools, the caches and the wire are all provable offline.

The scripted transport is NOT a test-only device. It is how you demo Sven with
no key, and how you replay a turn that went wrong from a saved transcript, which
is the only way to debug a model's behaviour without paying for it twice.

EVENTS, deliberately smaller than the SDK's own shape:

    {"type": "text",     "text": "..."}          a chunk of the answer
    {"type": "tool_use", "id":.., "name":.., "input": {...}}
    {"type": "end",      "stop_reason": "...", "usage": {...}}

Narrowing here rather than passing SDK objects upward means the loop never
touches a vendor type, and swapping or upgrading the SDK cannot reach past this
module.
"""

import json
import os

#: Default. Most turns are routing plus reading back a figure the Python already
#: computed, which is well inside Haiku, and it is the difference between a first
#: token that feels instant and one that feels like waiting.
FAST_MODEL = "claude-haiku-4-5-20251001"

#: Where `escalate` goes. Sonnet rather than Opus: the questions being escalated
#: are "read this evidence carefully", not "design a system", and Sonnet is the
#: cheapest model that is reliably careful with an interval.
DEEP_MODEL = "claude-sonnet-5"

MAX_TOKENS = 4000


class SvenUnavailable(RuntimeError):
    """Raised when the model cannot be reached, with the fix in the message.

    A distinct type because the CLI treats it differently from a bad answer:
    this is a setup problem the pilot can fix in one command, and it should read
    as such rather than as a stack trace.
    """


class ScriptedTurn:
    """Replays canned events. No network, no key, no SDK.

    `script` is either a list of turns (each a list of events) played in order,
    or a callable taking `(messages, tools)` and returning one turn's events —
    which is what lets a test assert on what the model was actually sent.
    """

    def __init__(self, script):
        self._script = script
        self._i = 0
        self.seen = []

    def stream(self, *, system, messages, tools, model):
        self.seen.append({"system": system, "messages": messages,
                          "tools": [t["name"] for t in tools], "model": model})
        if callable(self._script):
            events = self._script(messages, tools)
        else:
            if self._i >= len(self._script):
                raise AssertionError(
                    f"ScriptedTurn ran out of script after {self._i} turn(s) — "
                    f"the loop asked for one more than the test provided")
            events = self._script[self._i]
            self._i += 1
        yield from events


class AnthropicTurn:
    """The real thing. Imports the SDK lazily so the module is importable without it."""

    def __init__(self, api_key=None):
        try:
            import anthropic
        except ImportError as exc:
            raise SvenUnavailable(
                "Sven needs the Anthropic SDK, which the core install "
                "deliberately does not pull.\n"
                "  pip install -e \".[ask]\"\n"
                "Everything else still works — the deterministic commands are "
                "the same ones Sven would have called."
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SvenUnavailable(
                "No ANTHROPIC_API_KEY in the environment.\n"
                "  export ANTHROPIC_API_KEY=sk-...\n"
                "Sven is the only thing that needs it; every pilot command runs "
                "without one.")
        self._client = anthropic.Anthropic(api_key=key)

    def stream(self, *, system, messages, tools, model):
        """Yield narrowed events. Vendor objects never escape this method."""
        import anthropic

        try:
            with self._client.messages.stream(
                model=model, max_tokens=MAX_TOKENS, system=system,
                messages=messages, tools=tools,
            ) as stream:
                for event in stream.text_stream:
                    yield {"type": "text", "text": event}
                final = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise SvenUnavailable(
                f"Anthropic rejected the key: {exc}. Check ANTHROPIC_API_KEY."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise SvenUnavailable(f"Rate limited: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise SvenUnavailable(
                f"Could not reach the API ({exc}). The deterministic commands "
                f"still work offline — `manamap pilot deck-info <slug>`."
            ) from exc

        for block in final.content:
            if getattr(block, "type", None) == "tool_use":
                yield {"type": "tool_use", "id": block.id, "name": block.name,
                       "input": dict(block.input or {})}
        yield {"type": "end", "stop_reason": final.stop_reason,
               "usage": {"in": final.usage.input_tokens,
                         "out": final.usage.output_tokens}}


def default_turn():
    """The transport this environment should use.

    `MANAMAP_SVEN_TRANSPORT=scripted:<path.json>` swaps in a recorded script,
    which is how a demo runs with no key and how a bad turn is replayed.
    """
    spec = os.environ.get("MANAMAP_SVEN_TRANSPORT", "anthropic")
    if spec.startswith("scripted:"):
        with open(spec.split(":", 1)[1]) as f:
            return ScriptedTurn(json.load(f))
    return AnthropicTurn()
