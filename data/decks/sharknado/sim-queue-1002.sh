#!/bin/zsh
# Queued 2026-10-02/03. Strictly one heavy job at a time (the machine crashed twice
# with training and Forge together):
#   1. wait for the edgar A/A
#   2. retrain the ability model and rebuild everything downstream of it, then the
#      embedding quality gate (it read recall@10 0.230 against a 0.232 floor)
#   3. the sharknado champion, then momentum-v1, 200 games each on standard-v3
cd /Users/michellemacrae/mana-map
export MANAMAP_NO_DAEMON=1
while pgrep -f "experiment edgar-vampires" >/dev/null; do sleep 120; done
echo "$(date '+%F %T') A/A done"
for s in train-ability embed reduce export synergy power-creep cluster-regions card-roles viz-index eval-embeddings; do
  echo "$(date '+%F %T') STEP $s"
  .venv/bin/manamap $s || { echo "$(date '+%F %T') STEP $s FAILED exit=$?"; break; }
done
echo "$(date '+%F %T') GATE"
.venv/bin/pytest -n0 -q -p no:cacheprovider tests/test_embedding_quality.py 2>&1 | tail -15
for t in sharknado sharknado@momentum-v1; do
  echo "$(date '+%F %T') START $t"
  .venv/bin/manamap pilot simulate $t --pod standard-v3 --games 200 --jobs 4
  echo "$(date '+%F %T') END $t exit=$?"
done
