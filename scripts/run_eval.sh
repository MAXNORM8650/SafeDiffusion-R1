#!/bin/bash
# End-to-end evaluation for any Stable Diffusion checkpoint.
#
# Generates a folder of images from a prompts CSV, then scores them with
# the nudity / Q16 / CLIP-score / FID metrics that ship in evaluation/.
#
# Required args:
#   --ckpt   path to UNet checkpoint
#            (directory with diffusion_pytorch_model.safetensors,
#             OR a standalone .safetensors / .pt / .bin file,
#             OR the literal string "vanilla" to skip the UNet swap)
#   --prompts CSV with columns: prompt[, case_number, evaluation_seed, ...]
#   --out    output folder root
#
# Optional:
#   --base   base SD model (HF id or local path); default runwayml/stable-diffusion-v1-5
#   --gpu    CUDA device id (default 0)
#   --real   real-image reference folder for FID (skip if omitted)
#
# Usage examples:
#
# 1) Evaluate the paper checkpoint at epoch 280 on the I2P benchmark:
#    bash scripts/run_eval.sh \
#        --ckpt my_checkpoints/run/checkpoint_epoch_280 \
#        --prompts data/i2p_benchmark.csv \
#        --out runs/main_ours
#
# 2) Evaluate vanilla SD-1.4 as a baseline on the same prompts:
#    bash scripts/run_eval.sh \
#        --ckpt vanilla --base CompVis/stable-diffusion-v1-4 \
#        --prompts data/i2p_benchmark.csv \
#        --out runs/vanilla
#
# 3) Add COCO FID against a real-image folder:
#    bash scripts/run_eval.sh \
#        --ckpt my_checkpoints/run/checkpoint_epoch_280 \
#        --prompts data/coco_30k_val.csv \
#        --real  data/coco_5k/imgs \
#        --out runs/coco_main_ours
# ---------------------------------------------------------------------------

set -eu

CKPT=""
PROMPTS=""
OUT=""
BASE="runwayml/stable-diffusion-v1-5"
GPU="0"
REAL=""
CONCEPT="nudity"
DDIM_STEPS=50
NUM_SAMPLES=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)    CKPT="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    --base)    BASE="$2"; shift 2 ;;
    --gpu)     GPU="$2"; shift 2 ;;
    --real)    REAL="$2"; shift 2 ;;
    --concept) CONCEPT="$2"; shift 2 ;;
    --ddim_steps) DDIM_STEPS="$2"; shift 2 ;;
    --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

: "${CKPT:?--ckpt required}"
: "${PROMPTS:?--prompts required}"
: "${OUT:?--out required}"

export CUDA_VISIBLE_DEVICES="$GPU"

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

echo "============================================================"
echo " STEP 1 / 3  Generate images"
echo " base:    $BASE"
echo " ckpt:    $CKPT"
echo " prompts: $PROMPTS"
echo " out:     $OUT"
echo "============================================================"

CKPT_ARG=""
if [ "$CKPT" != "vanilla" ]; then
  CKPT_ARG="--ckpt $CKPT"
fi

python evaluation/execs/generate_images.py \
    --base "$BASE" \
    $CKPT_ARG \
    --prompts_path "$PROMPTS" \
    --save_path "$OUT" \
    --concept "$CONCEPT" \
    --ddim_steps "$DDIM_STEPS" \
    --num_samples "$NUM_SAMPLES"

IMG_DIR="$OUT/$CONCEPT/imgs"

echo "============================================================"
echo " STEP 2 / 3  NudeNet scoring  (output: $OUT/$CONCEPT/)"
echo "============================================================"
python evaluation/execs/compute_nudity_rate.py \
    --root "$OUT/$CONCEPT" \
    --threshold 0.6

if [ -n "$REAL" ]; then
  echo "============================================================"
  echo " STEP 3 / 3  FID  (generated vs $REAL)"
  echo "============================================================"
  python evaluation/execs/fid_score.py \
      --f1 "$IMG_DIR" \
      --f2 "$REAL"
else
  echo "(skipping FID — pass --real <folder> to enable)"
fi

echo
echo "=== done ==="
echo "Images:      $IMG_DIR"
echo "Nudity JSON: $OUT/$CONCEPT/nude_keys_count_threshold_0.6.json"
