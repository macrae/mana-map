#!/bin/zsh
# Queued 2026-10-03: after the sharknado queue (champion, momentum-v1), run
# edgar-vampires@draw-v1 at 200 games on standard-v3. The champion baseline at this
# exact harness is the 2026-10-02 n200 run (ov8c347642 / aif7c3b6a8 / tl9473cf35).
cd /Users/michellemacrae/mana-map
export MANAMAP_NO_DAEMON=1
while pgrep -f "sim-queue-1002.sh" >/dev/null; do sleep 120; done
echo "$(date '+%F %T') START edgar-vampires@draw-v1"
.venv/bin/manamap pilot simulate edgar-vampires@draw-v1 --pod standard-v3 --games 200 --jobs 4
echo "$(date '+%F %T') END edgar-vampires@draw-v1 exit=$?"
