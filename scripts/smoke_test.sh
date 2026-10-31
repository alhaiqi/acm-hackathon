#!/usr/bin/env bash
# Offline end-to-end check (no Bedrock, no downloads): fake GEO-Bench tasks + scripted model.
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d)
export GEO_BENCH_DIR=$T/geobench GOTRAIN_HOME=$T/home PYTHONPATH=$PWD
python tests/make_fake_geobench.py "$GEO_BENCH_DIR"
python -W ignore tests/test_agent_loop.py | grep -E "tool calls logged|AGENT LOOP OK"
echo "scratch dir: $T"
