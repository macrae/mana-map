# hapatra

Awaiting the locked decklist (Moxfield export format supported). Paste it as
`decklist.txt` in this directory — first card under a `Commander:` header
(or marked `*CMDR*`), sideboard under `SIDEBOARD:`.

Then run the v2 pipeline (see .claude/skills/write-manual):

```
manamap pilot fetch-deck hapatra
manamap pilot validate-deck hapatra
manamap pilot goldfish hapatra        # after curating goldfish_targets.json
```
