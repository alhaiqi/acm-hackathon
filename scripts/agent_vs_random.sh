#!/usr/bin/env bash
# Agent vs random search, same task, same wall-clock budget, VALID only.
#
#   nohup bash scripts/agent_vs_random.sh m-eurosat 60 > ~/gotrain_logs/agent_vs_random.out 2>&1 &
#
# Each method gets its own GOTRAIN_HOME holding links to the task's TRAIN/VALID cache only: no test file
# exists there, the agent runs with --valid-only (guard cancels finalize, no auto-finalize), and the real
# leaderboard is untouched. The two run one after the other so they never share the GPU.
# If the Brain side's prompt rewrite (B6) lands on origin/main, the experiment is stopped to free the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
TASK=${1:-m-eurosat}
BUDGET=${2:-60}
REAL=${GOTRAIN_HOME:-$HOME/gotrain_home}
X=$HOME/gotrain_experiments/agent_vs_random_$(date +%Y%m%d_%H%M)
git fetch -q && BASE=$(git rev-parse origin/main)

make_home() {
    mkdir -p "$1/cache/$TASK"
    for f in train_X.npy train_y.npy valid_X.npy valid_y.npy meta.json train_stats.json; do
        ln -s "$REAL/cache/$TASK/$f" "$1/cache/$TASK/$f"
    done
    cp "$REAL/timing.json" "$1/" 2>/dev/null || true
}
make_home "$X/agent"; make_home "$X/random"
echo "[$(date +%T)] experiment dir $X (no test file: $(ls "$X/agent/cache/$TASK" | grep -c test_) test_* links)"

b6_landed() {
    git fetch -q 2>/dev/null
    git log --oneline "$BASE..origin/main" -- gotrain/prompts.py | grep -q .
}

# run "$@" in the background with GOTRAIN_HOME=$1; stop everything under $X if B6 lands
phase() {
    local home=$1; shift
    GOTRAIN_HOME=$home "$@" &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
        if b6_landed; then
            echo "[$(date +%T)] B6 LANDED on origin/main: stopping the experiment to free the GPU for A8"
            kill $pid 2>/dev/null; pkill -f "$X" 2>/dev/null; sleep 5; pkill -9 -f "$X" 2>/dev/null
            touch "$X/STOPPED_FOR_B6"; return 1
        fi
        sleep 60
    done
}

echo "[$(date +%T)] 1/2 agent, $BUDGET min, valid only"
phase "$X/agent" python -u -W ignore -m gotrain.agent --tasks "$TASK" --budget-min "$BUDGET" --valid-only || exit 0
echo "[$(date +%T)] 2/2 random search, $BUDGET min"
phase "$X/random" python -u -W ignore scripts/random_search.py "$TASK" --budget-min "$BUDGET" || exit 0

echo "[$(date +%T)] compare"
python -W ignore scripts/compare_search.py "$TASK" agent="$X/agent" random="$X/random" | tee "$X/compare.txt"
ls "$X"/*/results/*.lock.json 2>/dev/null && echo "UNEXPECTED: a lock file exists" || echo "no finalize happened (no lock files)"
echo "[$(date +%T)] AGENT VS RANDOM DONE"
