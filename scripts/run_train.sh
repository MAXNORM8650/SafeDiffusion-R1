#!/bin/bash
# GRPO + steering-reward post-training for Stable Diffusion.
#
# Prerequisites (one-time setup):
#   1. HPSv2 sources + checkpoints (used by the NSFWv2 steering reward).
#        export HPSV2_PATH=/path/to/HPSv2
#        export HPS_CKPT_PATH=/path/to/hps_ckpt
#      hps_ckpt must contain:
#        - open_clip_pytorch_model.bin
#        - HPS_v2.1_compressed.pt
#
#   2. Login to wandb (optional, for live logging):
#        wandb login
#
# Usage:
#   bash scripts/run_train.sh
#
# Override any config field on the command line, for example:
#   bash scripts/run_train.sh --config.train.steering_alpha 0.7
#   bash scripts/run_train.sh --config.num_generations 8 --config.sample.batch_size 8
# ---------------------------------------------------------------------------

set -eu

# GPU set and number of processes (match the command you used during training).
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${NPROC:=8}"
: "${MASTER_PORT:=19001}"

# Required env vars for the NSFWv2 reward (loaded from HPSv2 weights).
: "${HPSV2_PATH:?need HPSV2_PATH pointing to the HPSv2 repo}"
: "${HPS_CKPT_PATH:?need HPS_CKPT_PATH pointing to a dir with HPSv2.1 weights}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

torchrun --nproc_per_node=${NPROC} --master_port ${MASTER_PORT} \
    fastvideo/train.py \
    --config config/base.py \
    --config.reward_fn nsfwv2 \
    --config.num_generations 16 \
    --config.sample.batch_size 4 \
    --config.train.batch_size 4 \
    --config.train.steering_alpha 0.5 \
    "$@"
