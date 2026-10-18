`sharknado.json` is the RECORDED answer of `GET /api/external/deck?deck_id=N381DOgjy9ZU`
for the pilot's own public Sharknado (sleeved v1.2.0), pulled 2026-10-17 right after its
first scan (DTI 31/96, Bracket 4) — the pilot's deck, not a redistribution of anyone's.
`fullAnalysis` is set to null: the bench never stores that HTML.

`sample.json` is a small hand-built document in the same shape for the two-card `ldeck`
fixture the unit tests drive; `test_pilot_deckcheck.py` holds its key sets to the
recorded document's, so the stand-in cannot drift from the real form.
