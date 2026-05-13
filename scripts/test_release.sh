#!/bin/bash
# Smoke test: pull a SafeDiffusion-R1 variant from the Hub, generate
# 3 images, NudeNet-score them, print PASS/FAIL. Useful as a CI step
# and as a quick sanity check after `git clone`ing this repo.
#
# Usage:
#   bash scripts/test_release.sh [scaled|compact|empty-positive]   (default: scaled)
# ---------------------------------------------------------------------------

set -eu
VARIANT="${1:-scaled}"
case "$VARIANT" in scaled|compact|empty-positive) ;; *) echo "unknown variant: $VARIANT"; exit 1 ;; esac

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

cat > $TMP/prompts.csv <<'EOF'
case_number,prompt,evaluation_seed
0,a beautiful sunset over the mountains,42
1,a cat sleeping on a couch,43
2,a portrait of a smiling woman,44
EOF

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

echo "============================================================"
echo " Smoke test: ItsMaxNorm/SafeDiffusion-R1 / $VARIANT"
echo "============================================================"
PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
bash scripts/run_eval.sh \
    --base ItsMaxNorm/SafeDiffusion-R1 --subfolder "$VARIANT" \
    --ckpt vanilla \
    --prompts $TMP/prompts.csv \
    --out $TMP/out \
    --ddim_steps 20 > $TMP/run.log 2>&1

N=$(ls $TMP/out/nudity/imgs 2>/dev/null | wc -l)
if [ "$N" -ge 3 ]; then
  echo "PASS — generated $N images via ItsMaxNorm/SafeDiffusion-R1 subfolder=$VARIANT"
  exit 0
else
  echo "FAIL — only $N images generated (expected 3)"
  echo "--- last 20 lines of run.log ---"
  tail -20 $TMP/run.log
  exit 1
fi
