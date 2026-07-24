---
name: rules-lookup
description: Query the local Magic comprehensive-rules DB — semantic search for discovering relevant rules, exact lookup for verifying citations. Also covers building/refreshing the rules DB when a new CR is released.
---

# Rules lookup

```bash
.venv/bin/manamap pilot query-rules "does storm copy targets" --k 8 --json   # semantic discovery
.venv/bin/manamap pilot lookup-rule 702.40a --json                            # exact fetch
```

**Semantic vs exact**: use `query-rules` to *discover* which rules govern an interaction (try several phrasings; keyword mechanics also have `glossary:<term>` chunks). Use `lookup-rule` to *verify or quote* — citation quotes must be copied verbatim from `lookup-rule` output, and the rules-checker only ever uses exact lookup. A miss suggests near-IDs ("601" → 601.1, 601.2, ...).

## Build / refresh the DB

```bash
.venv/bin/manamap pilot download-rules   # idempotent (sha256 sidecar)
.venv/bin/manamap pilot build-rules-db   # ~3.9K chunks, ~30s embed
```

When WotC publishes a new CR (each set release), `download-rules` 404s or fetches unchanged content: get the current TXT link from https://magic.wizards.com/en/rules, update `CR_RULES_URL` in `src/manamap/config.py`, re-run both commands. Rule IDs are stable across releases; saved stack artifacts record their `rules_version`.
