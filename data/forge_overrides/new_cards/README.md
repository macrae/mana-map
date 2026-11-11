# New card scripts — cards the installed Forge does not ship (2026-11-11)

The Forge build under `~/.mana-map/forge` is dated 2026-08-08, two months before Reality
Fracture (FRA, 2026-10-02). Two cards Edgar runs have no script there, so the seat audit
refused any Edgar run (`unfound 2`). Upgrading Forge would move the harness under every
calibrated table and break `data/forge_patches/PumpAi.java` (compiled against 2.0.14); the
pilot chose to hand-script the two cards instead.

A file here is a WHOLE script for a card the engine does not ship. `forge_pilot.install()`
adds it to `cardsfolder.zip` as a new entry, and REFUSES one the engine does ship (that is
an override, which belongs under `../cards/` and must be derived from the shipped script).
These files are part of the override fingerprint (`card_overrides.sha`), so a run made with
them never pools with one made without.

Each script is written from the card's oracle text in the syntax of shipped scripts that
carry the same clauses, and is proven with `forge-cast-check` before any arm:

- `e/edgar_ancient_bloodlord.txt`: death trigger as Cruel Celebrant's; the sacrifice cost
  as the one shipped `Sac<1/Creature.Other;Planeswalker.Other/…>`; an AIPreference to feed
  tokens and small creatures, as Indulgent Aristocrat's.
- `i/ingris_stingerquill.txt`: the attack trigger as Rage Forger's (`DamageSource$
  TriggeredAttacker`) aimed at each opponent; Cadet is the shipped `c_2_2_homunculus`
  token renamed — colourless 2/2, the right size and colour; its creature type
  (Homunculus, not Wizard Soldier) is the one approximation, and nothing either deck runs
  reads it.
