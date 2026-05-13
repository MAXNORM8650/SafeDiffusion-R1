# SafeDiffusion-R1: Online Reward Steering for Safe Diffusion Post-Training

<p align="center">
  <img src="figures/teaser.png" width="100%"/>
</p>

GRPO-based safety post-training for Stable Diffusion using a **closed-form,
CLIP-based steering reward**. No separately trained safety classifier, no
paired safe/unsafe image dataset, no inference-time intervention — the
safety prior is baked into the UNet weights.

---

## TL;DR — what the method does

For an NSFW prompt $p$, vanilla SD produces an image $x$ aligned to the
unsafe text embedding $z_p$. We instead reward the model against a
**steered** target $z_p + \alpha \cdot v_{\text{safe}}$, where
$v_{\text{safe}}$ is a single direction in CLIP-text space computed
once from a small set of (safe, unsafe) anchor phrases:

$$
v_{\text{safe}} \;=\; \overline{z_{\text{safe anchors}}} - \overline{z_{\text{unsafe anchors}}},
\qquad
r(x, p) \;=\; \cos\!\big(z_{\text{img}}(x),\; z_p + \alpha\, v_{\text{safe}}\big).
$$

GRPO post-training then nudges the UNet to satisfy this steered reward.
Because $v_{\text{safe}}$ is computed from a **frozen** CLIP encoder, the
target is stationary — the on-policy samples drift, but the anchor they're
regressed onto does not. This is what prevents the FID collapse that
plain GRPO suffers from on the same safety objective (FID 250 at 0%
nudity vs. ours at FID 48 with comparable safety).

<p align="center">
  <img src="figures/method.png" width="92%"/>
  <br/>
  <em>Method overview: a frozen CLIP encoder defines a safety direction
  v_safe from a small anchor set; GRPO uses the steered embedding as the
  reward target during diffusion post-training.</em>
</p>

## Headline results (vs.\ SD-v1.4 baseline)

| Benchmark | Baseline (SD-v1.4) | **Ours** | Δ |
|---|---|---|---|
| I2P inappropriate-content rate | 48.9% | **18.07%** | **−63%** |
| NudeNet detections (I2P, 4.7k prompts) | 646 | **15** | **−97.7%** |
| GenEval compositional accuracy | 42.08% | **47.83%** | **+5.75 pp** |
| MMA-Diffusion (1000-prompt benchmark, ASR↓) | 22.6% | **2.6%** | **8.7×** safer |
| SneakyPrompt RL (skip-rate↑, 200 prompts) | 37% | **89.5%** | model resists most prompts before any attack |

The safety gains **generalize to 7 OOD harm categories** (hate,
harassment, violence, self-harm, shocking, illegal-activity, sexual) even
though training only sees benign + nudity-style negatives.

<p align="center">
  <img src="figures/qualitative_results.png" width="100%"/>
  <br/>
  <em>Qualitative comparison: explicit prompts at inference time produce
  benign content from our model while vanilla SD generates NSFW imagery.</em>
</p>

## How the steering direction works geometrically

<p align="center">
  <img src="figures/steering_geometry.png" width="55%"/>
</p>

A held-out NSFW prompt sits in the unsafe sub-region of CLIP-text space;
adding $\alpha\, v_{\text{safe}}$ translates it across the safe/unsafe
boundary while preserving the prompt-conditional content. The
reward then anchors the on-policy samples to this translated target.

### Steering strength $\alpha$ is smooth and not knife-edge

We sweep $\alpha \in [0, 2]$ on 180 NSFW + 200 GenEval prompts and
record reward $r(\alpha) = \cos(z_{\text{img}}, z_{\text{text}} + \alpha\,
v_{\text{safe}})$:

<p align="center">
  <img src="figures/alpha_sensitivity.png" width="55%"/>
</p>

- **SAFE prompts** stay essentially flat for $\alpha \le 0.5$ — utility
  is preserved.
- **NSFW prompts** drop monotonically with diminishing returns past
  $\alpha \approx 0.7$.
- Any $\alpha \in [0.3, 0.7]$ gives comparable safety/utility — the
  reward is not knife-edge sensitive.

### UMAP visualisation of the steering effect

<p align="center">
  <img src="figures/umap_steering.png" width="100%"/>
  <br/>
  <em>UMAP of CLIP text embeddings before/after steering at several α
  values, and the safety score s = z · v_safe as a function of α for safe
  (blue) and unsafe (red) prompts. The redirection is consistent across
  synonym, keyword-minimal, and negation perturbations.</em>
</p>

---

## Repository layout

```
SafeDiffusion-R1/
├── pyproject.toml                          # Minimal dependencies
├── assets/CoProv2_captions.txt             # Default training prompt corpus
├── config/base.py                          # Training config (ml_collections)
├── fastvideo/
│   ├── train.py                            # Main GRPO training script
│   └── models/stable_diffusion/            # DDIM step + pipeline with logprob
├── rewards/
│   ├── inference_reward.py                 # NSFWv2 steering reward (CLIP + v_safe)
│   └── safety_classifier.py                # Builds the linear safety direction
├── vendor/HPSv2/                           # Vendored HPSv2 sources (no separate clone)
├── evaluation/
│   ├── execs/                              # Eval entry-point scripts (see table)
│   └── utils/                              # Helper modules + NudeNet ONNX (in-repo)
├── figures/                                # Paper / docs figures used in this README
└── scripts/
    ├── run_train.sh                        # Canonical training launch (torchrun)
    └── run_eval.sh                         # One-shot eval pipeline for any SD ckpt
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

The HPSv2 source code is **vendored** at `vendor/HPSv2/` — no separate
clone needed. (Override with `export HPSV2_PATH=/path/to/your/HPSv2` if
desired.)

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

Any field in `config/base.py` is overridable with `--config.<dotted.path>`.

### Key config knobs

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

### Outputs

- **Checkpoints**: `<config.checkpoint_dir>/checkpoint_epoch_{N}/diffusion_pytorch_model.safetensors`.
  Drop into a fresh `StableDiffusionPipeline` UNet to evaluate.
- **Per-epoch mean reward log**: appended line-by-line to `<config.reward_log_file>`.
- **Sampled images during training**: `<config.sample_image_dir>/image-*.jpg`.
- **wandb run**: under project `<config.wandb_project>`.

### Reward variants

| `--config.reward_fn` | Description | Extra deps |
|---|---|---|
| `nsfwv2` | The paper's steering reward. Closed-form $v_{\text{safe}}$ direction in HPSv2 CLIP space + cosine alignment to the steered target. | `HPS_CKPT_PATH` |
| `hpsv2` | Vanilla HPS-v2 alignment reward (no safety steering — used as the ablation baseline in our paper). | same |
| `hpsv3` | HPS-v3 reward (no safety steering). | `pip install hpsv3` |

## Evaluate any Stable Diffusion model

`scripts/run_eval.sh` is a one-shot wrapper for the end-to-end
safety / utility evaluation of **any** Stable-Diffusion-style checkpoint.

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

`--base` accepts a HuggingFace model id (e.g.
`runwayml/stable-diffusion-v1-5`) or a local snapshot directory. Shorthands
`1.4`, `2.1` map to the official CompVis / Stability hubs.

The wrapper produces:

```
<--out>/
└── <--concept>/        # default: nudity
    ├── imgs/                                 # one image per prompt
    ├── nudity_threshold_0.6.json             # per-image NudeNet detections
    └── nude_keys_count_threshold_0.6.json    # aggregate counts incl. `nude_images`
```

### Evaluation scripts (`evaluation/execs/`)

Each is a standalone CLI. They expect a folder of generated images
(naming: `<case_number>_<seed>.png`) and, where relevant, a CSV with
prompts to match by `case_number`.

| Script | What it measures | Quick example |
|---|---|---|
| `generate_images.py` | Generate from a prompts CSV using a chosen UNet. | `python evaluation/execs/generate_images.py --ckpt my_checkpoints/run/checkpoint_epoch_280 --prompts_path data/i2p.csv --save_path runs/main_ours` |
| `exp_generate_single_image.py` | Single-prompt sanity generation. | `python evaluation/execs/exp_generate_single_image.py --prompt "a nude person" --ckpt ...` |
| `compute_nudity_rate.py` | NudeNet per-class detection over a folder. | `python evaluation/execs/compute_nudity_rate.py --root runs/main_ours/nudity --threshold 0.6` |
| `imageclassify.py` | NSFW image classifier. | `python evaluation/execs/imageclassify.py --folder runs/main_ours` |
| `clip_score.py` | CLIP-score between images and prompts. | `python evaluation/execs/clip_score.py --folder runs/main_ours --prompts_path data/coco.csv` |
| `fid_score.py` | FID against a reference image folder. | `python evaluation/execs/fid_score.py --f1 runs/main_ours --f2 data/coco_5k/imgs` |
| `lpips_score.py` | LPIPS between two image folders. | `python evaluation/execs/lpips_score.py --folder1 runs/main_ours --folder2 runs/vanilla` |
| `style_loss.py` | VGG-19 style-drift between original and edited image sets. | `python evaluation/execs/style_loss.py --original_path runs/vanilla --edited_path runs/main_ours --promtps_path data/coco.csv` |
| `unet_difference_norm.py` | L2 norm between two UNet checkpoints. | `python evaluation/execs/unet_difference_norm.py --ckpt1 ... --ckpt2 ...` |
| `module_percentage.py` | Per-layer relative parameter change between checkpoints. | `python evaluation/execs/module_percentage.py --ckpt1 ... --ckpt2 ...` |
| `Q16/eval.py` | Q16 inappropriate-content binary classifier. | `python evaluation/execs/Q16/eval.py --folder runs/main_ours` |

### Typical evaluation flow

```bash
# 1. Generate images from a prompts CSV with your trained UNet
bash scripts/run_eval.sh \
    --ckpt my_checkpoints/run/checkpoint_epoch_280 \
    --prompts data/i2p_benchmark.csv \
    --out runs/main_ours

# 2. (Optional) CLIP-score on benign captions for utility
python evaluation/execs/clip_score.py \
    --folder runs/main_ours/nudity --prompts_path data/coco_30k_val.csv

# 3. (Optional) Q16 as a second-opinion safety detector
python evaluation/execs/Q16/eval.py --folder runs/main_ours/nudity/imgs
```

The NudeNet ONNX lives at `evaluation/utils/metrics/nudenet/best_new.onnx`
(in repo). Q16 prompt embeddings live at `evaluation/execs/Q16/data/`.

## What's vendored vs.\ what you supply

| Component | In-repo? | If not, where to get it |
|---|---|---|
| HPSv2 source code | ✅ `vendor/HPSv2/` | n/a |
| HPSv2 v2.1 checkpoints (`open_clip_pytorch_model.bin`, `HPS_v2.1_compressed.pt`) | ❌ (5.6 GB) | HPSv2 releases → drop in `hps_ckpt/`, point `HPS_CKPT_PATH` |
| NudeNet ONNX (`best_new.onnx`) | ✅ `evaluation/utils/metrics/nudenet/` | n/a |
| Q16 prompt embeddings | ✅ `evaluation/execs/Q16/data/` | n/a |
| Stable Diffusion base weights | ❌ | HuggingFace Hub (auto-downloaded by `diffusers` on first run) |
| I2P / COCO / SneakyPrompt / MMA datasets | ❌ | Public datasets — point CLI flags at local copies |

## Pretrained model release

The three main models from our paper are released as **full Diffusers
pipelines** (drop-in `StableDiffusionPipeline` — not bare UNet checkpoints)
at [https://huggingface.co/ItsMaxNorm/SafeDiffusion-R1](https://huggingface.co/ItsMaxNorm/SafeDiffusion-R1):

| Subfolder | Anchor set | Description | When to use |
|---|---|---|---|
| `scaled` | 25 safe + 20 unsafe | Main paper checkpoint (`geneval_negative_steringreward_8gpus_scale`, epoch 280). | Default headline numbers — best balance of MMA + I2P + GenEval. |
| `compact` | 5 safe + 3 unsafe | `steringreward_7gpus`, epoch 300. | Lowest MMA-Diffusion ASR (2.6%); use when adversarial robustness is the priority. |
| `empty-positive` | 0 safe + 3 unsafe | Ablation: no safe anchors. | Reference for understanding the role of positive anchors. |

### Inference in a few lines

```python
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionPipeline
import os, torch

# diffusers' `StableDiffusionPipeline.from_pretrained` doesn't natively
# accept `subfolder=` for the *full* pipeline (only for single
# components), so we snapshot just the variant we want and load it.
local_root = snapshot_download(
    "ItsMaxNorm/SafeDiffusion-R1",
    allow_patterns="scaled/*",           # or "compact/*" / "empty-positive/*"
)
pipe = StableDiffusionPipeline.from_pretrained(
    os.path.join(local_root, "scaled"),
    torch_dtype=torch.float16,
).to("cuda")
img = pipe("a photo of a cat sleeping on a couch").images[0]
img.save("out.png")
```

### One-line smoke test

To verify your environment can pull and run a released variant end-to-end:

```bash
bash scripts/test_release.sh scaled       # or compact / empty-positive
# → PASS — generated 3 images via ItsMaxNorm/SafeDiffusion-R1 subfolder=scaled
```

### Evaluate a released model with `scripts/run_eval.sh`

The same wrapper that evaluates your locally-trained checkpoints also
works directly against the HF Hub release — pass `--base` as the HF repo
id and `--subfolder` as the variant:

```bash
# Main paper checkpoint, I2P benchmark
bash scripts/run_eval.sh \
    --base ItsMaxNorm/SafeDiffusion-R1 --subfolder scaled \
    --ckpt vanilla \
    --prompts data/i2p_benchmark.csv \
    --out runs/scaled_i2p

# Compact variant, MMA-Diffusion adversarial prompts
bash scripts/run_eval.sh \
    --base ItsMaxNorm/SafeDiffusion-R1 --subfolder compact \
    --ckpt vanilla \
    --prompts data/mma_adv_prompts.csv \
    --out runs/compact_mma

# Compare against vanilla SD-1.4 on the same set (baseline row)
bash scripts/run_eval.sh \
    --base CompVis/stable-diffusion-v1-4 \
    --ckpt vanilla \
    --prompts data/i2p_benchmark.csv \
    --out runs/vanilla_i2p
```

The HF Hub will cache each variant once (~2 GB) on first load.

### Reproducing the paper's main checkpoint from scratch

```bash
# 8-GPU run, NSFWv2 steering, α=0.5, group size 16, 300 epochs, save every 20
bash scripts/run_train.sh \
    --config.num_epochs 300 \
    --config.save_freq 20 \
    --config.train.steering_alpha 0.5 \
    --config.num_generations 16 \
    --config.sample.batch_size 4 \
    --config.train.batch_size 4
```

Bare UNet checkpoints (one `diffusion_pytorch_model.safetensors` per training
run, in the format produced by `train.py`) are also kept at
[ItsMaxNorm/diffusion-p](https://huggingface.co/ItsMaxNorm/diffusion-p) for
reproducibility of ablation studies (incl. the `CoProv2_grpo` GRPO-only
baseline at `CoProv2_grpo/checkpoint_epoch_300`).

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{safediffusion_r1_2026,
  title  = {SafeDiffusion-R1: Online Reward Steering for Safe Diffusion Post-Training},
  author = {(authors)},
  booktitle = {(venue)},
  year   = {2026}
}
```

## License

This project is released under the terms of the `LICENSE` file. The
vendored HPSv2 source under `vendor/HPSv2/` is redistributed under its
original MIT license (see `vendor/HPSv2/LICENSE`).
