"""`_cli` under concurrency — a bug that was latent until something made calls in parallel.

`serve._cli` captures a command's output with `contextlib.redirect_stdout(buf)`.
`sys.stdout` is process-global and `ThreadingHTTPServer` runs one handler per
thread, so two overlapping `/api/cli` requests interleave like this:

    A enters   saves the real stdout, sets bufA
    B enters   saves bufA, sets bufB
    A prints   -> lands in bufB
    A exits    restores the real stdout
    B exits    restores bufA

Two distinct failures, both demonstrated below:

1. **Cross-talk.** A's output is returned as B's result. For Sven that means one
   deck's figures attributed to another deck's question — confidently, with no
   error anywhere. This is the failure this repo least tolerates.

2. **A permanently dead stdout.** After that interleaving `sys.stdout` is a
   `StringIO` nobody holds, so every later `print` in the process vanishes. In a
   CLI that exits in seconds it is invisible; in `manamap serve` it is permanent
   until someone restarts it, and the symptom is "the server stopped logging".

It was unreachable in practice while `/api/cli` was called one request at a time.
Sven makes 3-8 tool calls a turn, so it became reachable the moment he existed.

The fix is a module-level lock around the redirect in `serve._cli`, which
serializes warm read-only commands. That is the right trade and it is worth
stating rather than hiding: those commands are 0.15-1.25 s warm, and correctness
of attribution is not negotiable.
"""

import contextlib
import io
import sys
import threading
import time

from manamap import serve


def test_redirect_stdout_is_process_global_not_per_thread():
    """The mechanism, pinned. If a future Python makes stdout thread-local this
    fails, and the lock in `_cli` can go."""
    real = sys.stdout
    a_in, b_in = threading.Event(), threading.Event()
    got = {}

    def a():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            a_in.set()
            b_in.wait(2)
            print("FROM-A")
        got["A"] = buf.getvalue()

    def b():
        buf = io.StringIO()
        a_in.wait(2)
        with contextlib.redirect_stdout(buf):
            b_in.set()
            time.sleep(0.2)
        got["B"] = buf.getvalue()

    ta, tb = threading.Thread(target=a), threading.Thread(target=b)
    ta.start(); tb.start(); ta.join(); tb.join()
    sys.stdout = real                      # repair what the demonstration broke

    assert "FROM-A" in got["B"], "expected the cross-talk this test documents"
    assert got["A"] == "", "A should have captured nothing — its output went to B"


def test_concurrent_cli_calls_do_not_swap_their_output():
    """The real thing, through `serve._cli`, with two different decks.

    Re-introduce the bug by removing the lock from `_cli` and this fails with one
    deck's report returned for the other, or with an empty result.
    """
    slugs = ["heliod", "gishath"]
    results = {}
    errors = []
    barrier = threading.Barrier(len(slugs))

    def call(slug):
        try:
            barrier.wait(5)                # maximise overlap
            results[slug] = serve._cli(["deck-status", slug])["stdout"]
        except Exception as exc:           # noqa: BLE001
            errors.append(f"{slug}: {exc.__class__.__name__}: {exc}")

    threads = [threading.Thread(target=call, args=(s,)) for s in slugs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)

    assert not errors, errors
    assert len(results) == len(slugs)
    for slug, out in results.items():
        assert out.strip(), f"{slug}: empty output — its stdout went to the other thread"
        assert slug in out, f"{slug}: got another deck's report:\n{out[:300]}"
    assert results["heliod"] != results["gishath"], "both threads returned one deck's output"


def test_stdout_survives_concurrent_cli_calls():
    """The second failure: a process whose stdout is left pointing at a dead buffer.

    Asserted separately because it outlives the request that caused it — the
    server goes quiet and nothing says why.
    """
    real = sys.stdout
    threads = [threading.Thread(target=lambda s=s: serve._cli(["deck-status", s]))
               for s in ("heliod", "gishath", "ur-dragon")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert sys.stdout is real, (
        f"sys.stdout was left as {type(sys.stdout).__name__}; every later print "
        f"in this process would vanish")
