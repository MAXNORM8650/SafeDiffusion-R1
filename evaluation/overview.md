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
