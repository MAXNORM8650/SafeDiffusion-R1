# SafeDiffusion-R1: Online Reward Steering for Safe Diffusion Post-Training

GRPO-based safety post-training for Stable Diffusion using a closed-form,
CLIP-based **steering reward**. No separately trained safety classifier,
no paired safe/unsafe image dataset.

## Repository layout

```
SteeringDiffusion/
├── pyproject.toml                          # Minimal dependencies
├── assets/CoProv2_captions.txt             # Default training prompt corpus
├── config/base.py                          # Training config (ml_collections)
├── fastvideo/
│   ├── train.py                            # Main GRPO training script
│   └── models/stable_diffusion/            # DDIM step + pipeline with logprob
├── rewards/
│   ├── inference_reward.py                 # NSFWv2 steering reward (CLIP + v_safe)
│   └── safety_classifier.py                # Builds the linear safety direction
├── vendor/HPSv2/                           # Vendored HPSv2 sources (used by NSFWv2)
├── evaluation/
│   ├── execs/                              # Eval entry-point scripts (see table below)
│   └── utils/                              # Helper modules + NudeNet ONNX
│       └── metrics/                        # nudity_eval, q16, clip_score, style_eval, ...
└── scripts/run_train.sh                    # Canonical launch script (torchrun)
```

## Setup

```bash
# 1. Install (editable).
pip install -e .

# 2. Drop the HPSv2 v2.1 weights somewhere (≈5.6 GB total):
mkdir -p hps_ckpt
# Download into hps_ckpt/:
#   open_clip_pytorch_model.bin
#   HPS_v2.1_compressed.pt
export HPS_CKPT_PATH=$(pwd)/hps_ckpt
```

The HPSv2 source code is already **vendored** under `vendor/HPSv2/` — you do
not need a separate clone. (If you want to override, set
`export HPSV2_PATH=/path/to/your/HPSv2`.)

## Train

The canonical launch (NSFWv2 steering reward, edit GPU count for your machine):

```bash
bash scripts/run_train.sh
```

Underneath, this runs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port 19001 \
    fastvideo/train.py \
    --config config/base.py \
    --config.reward_fn nsfwv2 \
    --config.num_generations 16 \
    --config.sample.batch_size 4 \
    --config.train.batch_size 4 \
    --config.train.steering_alpha 0.5
```

Override at invocation:

```bash
CUDA_VISIBLE_DEVICES=0,2 NPROC=2 bash scripts/run_train.sh        # 2 GPUs
bash scripts/run_train.sh --config.train.steering_alpha 0.7       # tune α
bash scripts/run_train.sh --config.num_generations 8              # group size
```

Any field in `config/base.py` can be overridden with `--config.<dotted.path>`
on the command line.

## Key config knobs

| Field | Default | Purpose |
|---|---|---|
| `config.pretrained.model` | `runwayml/stable-diffusion-v1-5` | Base diffusion checkpoint. |
| `config.reward_fn` | placeholder (pass `nsfwv2` on CLI) | Reward: `nsfwv2`, `hpsv2`, `hpsv3`. |
| `config.train.steering_alpha` | `0.5` | NSFWv2 steering strength (paper sweet spot: 0.5). |
| `config.num_generations` | `4` | Group size in GRPO. |
| `config.sample.batch_size` / `config.train.batch_size` | `1` / `1` | Per-GPU sampling / training batch sizes. |
| `config.num_epochs` | `300` | Training length. |
| `config.save_freq` | `20` | Save a UNet checkpoint every N epochs. |
| `config.prompt_file` | `assets/CoProv2_captions.txt` | Newline-delimited training prompts. |
| `config.checkpoint_dir` | `./my_checkpoints/run` | Where epoch checkpoints are written. |
| `config.sample_image_dir` | `./samples` | Where generated images are dumped during training. |
| `config.reward_log_file` | `./reward_per_epoch.txt` | Per-epoch mean-reward log. |
| `config.wandb_project` | `steering-diffusion-grpo` | wandb project name. |

> All artefact paths default to **relative** locations, so running
> `bash scripts/run_train.sh` from the repo root drops checkpoints,
> samples, and the reward log under the repo. Override with
> `--config.checkpoint_dir /absolute/path` to dump elsewhere.

## Outputs

- **Checkpoints**: `<config.checkpoint_dir>/checkpoint_epoch_{N}/diffusion_pytorch_model.safetensors`.
  Drop this into a fresh `StableDiffusionPipeline` UNet to evaluate.
- **Per-epoch mean reward log**: appended line-by-line to `<config.reward_log_file>`.
- **Sampled images during training**: `<config.sample_image_dir>/image-*.jpg`.
- **wandb run**: under project `<config.wandb_project>`.

## Reward variants

| `--config.reward_fn` | Description | Extra deps |
|---|---|---|
| `nsfwv2` | The paper's steering reward. Computes a `v_safe` direction in HPSv2 CLIP space from a small anchor set, then rewards `cos(z_image, z_text + α·v_safe)`. | `HPS_CKPT_PATH` |
| `hpsv2` | Vanilla HPS-v2 alignment reward (no safety steering). | same |
| `hpsv3` | HPS-v3 reward (no safety steering). | `pip install hpsv3` |

## Evaluate any Stable Diffusion model

`scripts/run_eval.sh` is a one-shot wrapper that takes **any** SD-style
diffusion model (your trained checkpoint or a vanilla baseline) and runs
the end-to-end safety / utility evaluation.

```bash
# 1) Your trained checkpoint (epoch 280)
bash scripts/run_eval.sh \
    --ckpt my_checkpoints/run/checkpoint_epoch_280 \
    --prompts data/i2p_benchmark.csv \
    --out runs/main_ours

# 2) Vanilla SD-1.4 baseline (no UNet swap)
bash scripts/run_eval.sh \
    --ckpt vanilla --base CompVis/stable-diffusion-v1-4 \
    --prompts data/i2p_benchmark.csv \
    --out runs/vanilla

# 3) Any HuggingFace SD model + COCO FID
bash scripts/run_eval.sh \
    --ckpt vanilla --base stabilityai/stable-diffusion-2-1-base \
    --prompts data/coco_30k_val.csv \
    --real  data/coco_5k/imgs \
    --out runs/sd21_coco
```

`--ckpt` accepts three forms:

| Form | Example | What it does |
|---|---|---|
| **directory** | `my_checkpoints/run/checkpoint_epoch_280` | `UNet2DConditionModel.from_pretrained(...)` — the natural output of `train.py` |
| **`.safetensors` file** | `path/to/diffusion_pytorch_model.safetensors` | loads as a state-dict into the base SD UNet |
| **`vanilla`** | literal string | skips the UNet swap, uses the `--base` model as-is |

`--base` accepts either a HuggingFace model id (e.g.
`runwayml/stable-diffusion-v1-5`) or a local snapshot directory.
Shorthands `1.4`, `2.1` are mapped to the official CompVis / Stability hubs.

The wrapper produces:

```
<--out>/
└── <--concept>/        # default: nudity
    ├── imgs/                                 # one image per prompt
    ├── nudity_threshold_0.6.json             # per-image NudeNet detections
    └── nude_keys_count_threshold_0.6.json    # aggregate counts incl. `nude_images`
```

## Evaluation scripts (`evaluation/execs/`)

Each script is a standalone CLI. They expect a folder of generated images
(naming convention: `<case_number>_<seed>.png`) and, where relevant, a CSV
with prompts to match by `case_number`.

| Script | What it measures | Quick example |
|---|---|---|
| `generate_images.py` | Generate a folder of images from a prompts CSV using a chosen UNet. | `python evaluation/execs/generate_images.py --ckpt my_checkpoints/run/checkpoint_epoch_280/diffusion_pytorch_model.safetensors --prompts_path data/i2p.csv --save_path runs/main_ours` |
| `exp_generate_single_image.py` | Single-prompt sanity generation. | `python evaluation/execs/exp_generate_single_image.py --prompt "a nude person" --ckpt ...` |
| `compute_nudity_rate.py` | NudeNet per-class detection over a folder; outputs a JSON of per-image labels and aggregate `nude_keys_count_threshold_<t>.json`. | `python evaluation/execs/compute_nudity_rate.py --root runs/main_ours/nudity --threshold 0.6` |
| `imageclassify.py` | NSFW image classifier (Falconsai-style). | `python evaluation/execs/imageclassify.py --folder runs/main_ours` |
| `clip_score.py` | CLIP-score between generated images and the prompt set. | `python evaluation/execs/clip_score.py --folder runs/main_ours --prompts_path data/coco.csv` |
| `fid_score.py` | FID against a reference image folder. | `python evaluation/execs/fid_score.py --f1 runs/main_ours --f2 data/coco_5k/imgs` |
| `lpips_score.py` | LPIPS between two image folders (pairwise by filename). | `python evaluation/execs/lpips_score.py --folder1 runs/main_ours --folder2 runs/vanilla` |
| `style_loss.py` | VGG-19 style-drift between original and edited image sets. | `python evaluation/execs/style_loss.py --original_path runs/vanilla --edited_path runs/main_ours --promtps_path data/coco.csv` |
| `unet_difference_norm.py` | L2 norm between two UNet checkpoints (sanity check of how far training drifted). | `python evaluation/execs/unet_difference_norm.py --ckpt1 ... --ckpt2 ...` |
| `module_percentage.py` | Per-layer relative parameter change between checkpoints. | `python evaluation/execs/module_percentage.py --ckpt1 ... --ckpt2 ...` |
| `Q16/eval.py` | Q16 inappropriate-content binary classifier. | `python evaluation/execs/Q16/eval.py --folder runs/main_ours` |

### Typical evaluation flow

```bash
# 1. Generate images from a prompts CSV with your trained UNet
python evaluation/execs/generate_images.py \
    --ckpt my_checkpoints/run/checkpoint_epoch_280/diffusion_pytorch_model.safetensors \
    --prompts_path data/i2p_benchmark.csv \
    --save_path runs/main_ours --concept nudity

# 2. Score nudity (writes nude_keys_count_threshold_0.6.json into the same dir)
python evaluation/execs/compute_nudity_rate.py \
    --root runs/main_ours/nudity --threshold 0.6

# 3. Compute COCO FID against a real-image reference folder
python evaluation/execs/fid_score.py \
    --f1 runs/coco_main_ours --f2 data/coco_5k/imgs

# 4. CLIP-score (text-image alignment) on benign captions
python evaluation/execs/clip_score.py \
    --folder runs/coco_main_ours --prompts_path data/coco_30k_val.csv
```

The NudeNet ONNX model lives at `evaluation/utils/metrics/nudenet/best_new.onnx`
(already in repo). The Q16 prompt embeddings live at
`evaluation/execs/Q16/data/`.

## What's vendored vs.\ what you supply

| Component | In-repo? | If not, where to get it |
|---|---|---|
| HPSv2 source code | ✅ `vendor/HPSv2/` | n/a |
| HPSv2 v2.1 checkpoint (`open_clip_pytorch_model.bin`, `HPS_v2.1_compressed.pt`) | ❌ (5.6 GB) | HPSv2 GitHub releases → put in `hps_ckpt/`, point `HPS_CKPT_PATH` at it |
| NudeNet ONNX (`best_new.onnx`) | ✅ `evaluation/utils/metrics/nudenet/` | n/a |
| Q16 prompt embeddings | ✅ `evaluation/execs/Q16/data/` | n/a |
| Stable Diffusion base weights | ❌ | HuggingFace Hub (auto-downloaded by `diffusers` on first run) |
| I2P / COCO / SneakyPrompt / MMA datasets | ❌ | Public datasets — point CLI flags at your local copies |
