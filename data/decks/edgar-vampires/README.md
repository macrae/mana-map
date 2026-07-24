# edgar-vampires

Awaiting the locked decklist (Moxfield export format supported). Paste it as
`decklist.txt` in this directory — first card under a `Commander:` header
(or marked `*CMDR*`), sideboard under `SIDEBOARD:`.

Then run the v2 pipeline (see .claude/skills/write-manual):

```
manamap pilot fetch-deck edgar-vampires
manamap pilot validate-deck edgar-vampires
manamap pilot goldfish edgar-vampires        # after curating goldfish_targets.json
```
