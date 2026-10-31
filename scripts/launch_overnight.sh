#!/usr/bin/env bash
# A8: the overnight agent runs on the two tasks whose test split is still unused.
# Run it AFTER the Brain side's B6 (prompt rewrite) is on main (sync point 3).
#
#   nohup bash scripts/launch_overnight.sh > ~/gotrain_logs/overnight.out 2>&1 &
#
# Budgets come from the measured timings (T4, resnet18, full data, seconds per epoch):
#   m-brick-kiln   4.5 s/epoch -> a 30-epoch run ~2 min: 100 min is ~40 full runs, plenty for the tournament
#   m-chesapeake  31.75 s/epoch -> a 30-epoch run ~16 min (larger models ~2x): 150 min fits screen + race + branches
# Override: BUDGET_BRICK=.. BUDGET_CHES=.. bash scripts/launch_overnight.sh
set -uo pipefail
cd "$(dirname "$0")/.."
BUDGET_BRICK=${BUDGET_BRICK:-100}
BUDGET_CHES=${BUDGET_CHES:-150}
HOME_DIR=${GOTRAIN_HOME:-$HOME/gotrain_home}

echo "[$(date +%T)] waiting for other agent runs to finish"
while pgrep -f "gotrain.agent" >/dev/null; do sleep 30; done

git pull -q && git log --oneline -1 | cat
pip install -q -r requirements.txt >/dev/null 2>&1

echo "[$(date +%T)] smoke test (must pass before any test split is spent)"
if ! bash scripts/smoke_test.sh | tee /tmp/overnight_smoke.txt | grep -q "ENGINE OK"; then
    echo "SMOKE TEST FAILED - not launching"; cat /tmp/overnight_smoke.txt; exit 1
fi
grep -q "AGENT LOOP OK" /tmp/overnight_smoke.txt || { echo "AGENT LOOP FAILED - not launching"; exit 1; }

for spec in "m-brick-kiln:$BUDGET_BRICK" "m-chesapeake:$BUDGET_CHES"; do
    task=${spec%%:*}; budget=${spec##*:}
    if [ -e "$HOME_DIR/results/$task.lock.json" ]; then
        echo "[$(date +%T)] $task: test split already used - skipped"; continue
    fi
    echo "[$(date +%T)] $task: agent with $budget min${GOTRAIN_S3_BUCKET:+ (store: s3://$GOTRAIN_S3_BUCKET/${GOTRAIN_S3_PREFIX:-gotrain})}"
    python -u -W ignore -m gotrain.agent --tasks "$task" --budget-min "$budget"
done
python -W ignore -m gotrain.report
echo "[$(date +%T)] OVERNIGHT DONE"
