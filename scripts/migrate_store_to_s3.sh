#!/usr/bin/env bash
# A7: move the local model/dataset store to S3 once RASID gives the team a bucket and prefix.
#
#   GOTRAIN_S3_BUCKET=<bucket> GOTRAIN_S3_PREFIX=<prefix> bash scripts/migrate_store_to_s3.sh
#
# The local store already has the S3 layout (store.py), so this is a copy plus a check that the
# registry read back FROM S3 matches the local one. Afterwards keep the two variables exported (or put
# them in ~/.bashrc) so new runs publish straight to S3.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GOTRAIN_S3_BUCKET:?set GOTRAIN_S3_BUCKET to the bucket RASID gave the team}"
PREFIX=${GOTRAIN_S3_PREFIX:-gotrain}
STORE=${GOTRAIN_HOME:-$HOME/gotrain_home}/store
DEST="s3://$GOTRAIN_S3_BUCKET/$PREFIX"

echo "write check: $DEST"
echo "gotrain write check $(date)" > /tmp/gotrain_write_check.txt
aws s3 cp /tmp/gotrain_write_check.txt "$DEST/_write_check.txt" --only-show-errors
aws s3 rm "$DEST/_write_check.txt" --only-show-errors

echo "copy $STORE -> $DEST"
aws s3 sync "$STORE" "$DEST/" --only-show-errors
aws s3 ls "$DEST/" --recursive --summarize | tail -2

echo "verify: registry read from S3 == local registry"
GOTRAIN_S3_PREFIX=$PREFIX python -W ignore - "$STORE" <<'EOF'
import json, sys
from pathlib import Path
from gotrain.store import Store
st, ok = Store(), True
for reg in sorted(Path(sys.argv[1]).glob("models/*/registry.json")):
    task = reg.parent.name
    local, remote = json.loads(reg.read_text()), st.versions(task)
    same = [v["run_id"] for v in local] == [v["run_id"] for v in remote]
    ok &= same
    print(f"  {task}: {len(remote)} versions in S3, stages {[v['stage'] for v in remote]} {'OK' if same else 'MISMATCH'}")
sys.exit(0 if ok else 1)
EOF
echo "S3 MIGRATION OK - now: export GOTRAIN_S3_BUCKET=$GOTRAIN_S3_BUCKET GOTRAIN_S3_PREFIX=$PREFIX"
