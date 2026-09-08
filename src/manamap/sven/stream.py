"""The wire between the daemon and `mm ask`, and the rule it exists to keep.

`console.py` states the rule this repo already lives by:

    stdout is the ANSWER; stderr is the theatre.

So `mm ask --json | jq` has to stay byte-clean while Sven narrates. A single
stream of text cannot do that — narration and answer would interleave into one
pipe and the theatre would end up in the data. Hence typed frames: the client
routes `text` to stdout and everything else to stderr, and the split survives
whatever the model decides to say.

Server-sent events rather than a bespoke framing because it is a line protocol
with an existing shape, it survives an HTTP proxy, and `data:` lines make a
partial read obvious instead of silently truncating a JSON document.

    event: text     one chunk of the answer            -> stdout
    event: tool     "reading zur's gates"              -> stderr
    event: note     a caveat or a cost quote           -> stderr
    event: error    something failed, in words         -> stderr
    event: done     the turn's summary and stats       -> stderr

FRAMES ARE ORDERED AND NEVER MERGED. A `text` frame is written the moment it
arrives so the first token lands early; buffering here would undo the whole
reason for streaming.
"""

import json

#: Frame kinds. `text` is the only one that reaches stdout — a new kind is
#: theatre by default, which is the safe direction for a rule about data purity.
KINDS = ("text", "tool", "note", "error", "done")

STDOUT_KINDS = frozenset({"text"})


def encode(kind, data):
    """One SSE frame. `data` is JSON so a newline in the payload cannot split it."""
    if kind not in KINDS:
        raise ValueError(f"stream: {kind!r} is not one of {KINDS}")
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def decode(chunk):
    """Parse one SSE frame into `(kind, data)`, or `None` if it is not one.

    Tolerant by design: a keep-alive comment or a blank line is not an error,
    and a malformed frame must not take down a turn that is otherwise fine.
    """
    kind = None
    payload = None
    for line in chunk.splitlines():
        if line.startswith("event:"):
            kind = line[6:].strip()
        elif line.startswith("data:"):
            payload = line[5:].strip()
    if kind not in KINDS or payload is None:
        return None
    try:
        return kind, json.loads(payload)
    except ValueError:
        return None


def iter_frames(lines):
    """Yield `(kind, data)` from an iterable of decoded response lines.

    Accumulates until a blank line closes a frame, which is what SSE specifies
    and what lets a `data:` payload be arbitrarily long.
    """
    buf = []
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        line = line.rstrip("\n")
        if line == "":
            frame = decode("\n".join(buf))
            buf = []
            if frame:
                yield frame
            continue
        buf.append(line)
    if buf:
        frame = decode("\n".join(buf))
        if frame:
            yield frame


def route(kind, data, out, err):
    """Write one frame to the correct stream. The whole point of the module.

    Returns True if anything reached stdout, so a caller can tell "the model
    said nothing" from "the model said something and we dropped it".
    """
    if kind in STDOUT_KINDS:
        out.write(data if isinstance(data, str) else json.dumps(data))
        out.flush()
        return True
    if kind == "tool":
        err.write(f"  · {data}\n")
    elif kind == "note":
        err.write(f"  {data}\n")
    elif kind == "error":
        err.write(f"  ! {data}\n")
    elif kind == "done" and isinstance(data, dict) and data.get("summary"):
        err.write(f"  {data['summary']}\n")
    err.flush()
    return False
