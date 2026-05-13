"""
Simple Inference Reward: Just give image + prompt → get reward

Logic:
- If text is negative → steer it to positive, then compute alignment
- If text is positive → normal alignment
"""

import os
import torch
from PIL import Image
import sys

# Resolve in-repo HPSv2 sources (vendored at <repo>/vendor/HPSv2) by default.
# Override with HPSV2_PATH env var to point at a different clone.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_HPSV2 = os.path.join(_REPO_ROOT, "vendor", "HPSv2")
sys.path.insert(0, os.environ.get('HPSV2_PATH', _DEFAULT_HPSV2))

from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from .safety_classifier import SafetySteering

device = "cuda" if torch.cuda.is_available() else "cpu"
_CLASSIFIER = None

def initialize():
    """Initialize model once"""
    global _CLASSIFIER
    if _CLASSIFIER is not None:
        return

    arch = "ViT-H-14"
    model, _, preprocess = create_model_and_transforms(
        arch, os.environ.get('HPS_CKPT_PATH', 'hps_ckpt') + '/open_clip_pytorch_model.bin',
        precision="amp", device=device, jit=False,
        force_quick_gelu=False, force_custom_text=False,
        force_patch_dropout=False, output_dict=True,
        with_score_predictor=False, with_region_predictor=False,
    )

    checkpoint = torch.load(os.environ.get('HPS_CKPT_PATH', 'hps_ckpt') + '/HPS_v2.1_compressed.pt', map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)

    tokenizer = get_tokenizer(arch)
    _CLASSIFIER = SafetySteering(model.to(device).eval(), tokenizer, preprocess, device)
    _CLASSIFIER.build_safety_direction()
    print("✓ Reward model ready")


def get_reward(image, prompt, steering_alpha=0.5, return_tensor=False):
    """
    Get reward for image + prompt

    Automatically steers negative text to positive

    Args:
        image: PIL Image, torch.Tensor, or path
        prompt: text string
        steering_alpha: how much to steer negative text (0.3-0.7)
        return_tensor: if True, return torch.Tensor; if False, return float

    Returns:
        reward: float or torch.Tensor (higher is better)
    """
    if _CLASSIFIER is None:
        initialize()

    # Convert image if needed
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, torch.Tensor):
        image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))

    # Get text safety score
    z_text = _CLASSIFIER.encode_text([prompt])
    text_safety = (z_text * _CLASSIFIER.v_safe).sum(dim=-1)

    # Get image embedding
    z_image = _CLASSIFIER.encode_image(image)
    
    # If text is negative, steer it to positive
    z_text_steered = _CLASSIFIER.steer_text_toward_safe(prompt, alpha=steering_alpha)
    reward = (z_image * z_text_steered).sum(dim=-1)

    if return_tensor:
        return reward  # torch.Tensor with shape [1]
    else:
        return reward.item()  # float


def get_batch_rewards(images, prompts, steering_alpha=0.5):
    """
    Get rewards for batch of images and prompts

    Args:
        images: list of PIL Images or torch.Tensors
        prompts: list of text strings
        steering_alpha: steering strength

    Returns:
        rewards: torch.Tensor of shape [batch_size]
    """
    rewards = []
    for image, prompt in zip(images, prompts):
        reward = get_reward(image, prompt, steering_alpha, return_tensor=True)
        rewards.append(reward)

    return torch.cat(rewards, dim=0)


# Example usage
if __name__ == "__main__":
    # Test
    image = "sde_samples_neg_ddim/trajectory_00.png"

    prompts = [
        "a beautiful landscape photo",
        "a photo of nude woman",
        "explicit sexual content",
    ]

    print("Image + Prompt → Reward\n")
    for prompt in prompts:
        reward = get_reward(image, prompt)
        print(f"{prompt:40s} → {reward:+.4f}")
