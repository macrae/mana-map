"""A producer for `test_pilot_deck_edit`'s pool test: importable by NAME in a
spawned worker (a monkeypatched stub never reaches one), cheap, deterministic.

`main(args)` writes `probe-<n>.txt` into the deck's directory, after a sleep that
makes the LAST job finish FIRST — so a pool that returned results in completion
order would be caught. `n < 0` raises, to prove a failure is a result."""
import time

from manamap.pilot.common import deck_dir


def main(args):
    if args.n < 0:
        raise ValueError(f"probe {args.n} refuses")
    time.sleep(0.05 * (4 - args.n))
    text = "".join(f"{args.slug}:{args.n}:{i * i}\n" for i in range(100))
    (deck_dir(args.slug) / f"probe-{args.n}.txt").write_text(text)
    print(f"probe {args.n} wrote")
