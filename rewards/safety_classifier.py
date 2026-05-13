"""
4-Category Safety Classifier using Steering Vectors
Categories:
1. Both image & text are safe
2. Image is unsafe, text is safe
3. Image is safe, text is unsafe
4. Both image & text are unsafe
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from PIL import Image

class SafetySteering:
    def __init__(self, model, tokenizer, preprocess, device="cuda"):
        """
        Args:
            model: HPSv2 or OpenCLIP model
            tokenizer: Text tokenizer
            preprocess: Image preprocessor
            device: cuda or cpu
        """
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = device
        self.v_safe = None

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text to normalized embeddings"""
        tokens = self.tokenizer(texts).to(self.device)
        z = self.model.encode_text(tokens)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def encode_image(self, images) -> torch.Tensor:
        """Encode images to normalized embeddings"""
        if isinstance(images, Image.Image):
            images = [images]
        x = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        z = self.model.encode_image(x)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def build_safety_direction(
        self,
        safe_texts: List[str] = None,
        unsafe_texts: List[str] = None
    ) -> torch.Tensor:
        """
        Build safety direction vector from safe/unsafe text pairs

        Returns:
            v_safe: normalized safety direction vector
        """
        if safe_texts is None:
            safe_texts = [
                "a safe",
                "a non-violent, non-sexual image",
                "a wholesome, PG-rated photo",
                "an image with no nudity, no gore, no weapons",
                "appropriate image",
            ]

        if unsafe_texts is None:
            unsafe_texts = [
                "explicit sexual content and nudity",
                "pornographic material",
                "an image with naked people engaged in sexual acts",
            ]

        z_safe = self.encode_text(safe_texts).mean(dim=0, keepdim=True)
        z_unsafe = self.encode_text(unsafe_texts).mean(dim=0, keepdim=True)
        v = z_safe - z_unsafe
        self.v_safe = F.normalize(v, dim=-1)
        return self.v_safe

    @torch.no_grad()
    def compute_safety_scores(
        self,
        image,
        text: str
    ) -> Dict[str, float]:
        """
        Compute safety scores for image and text

        Returns:
            dict with:
                - image_safety: float [-1, 1]
                - text_safety: float [-1, 1]
                - alignment: cos(image, text) for prompt alignment
        """
        if self.v_safe is None:
            raise ValueError("Must call build_safety_direction() first!")

        z_image = self.encode_image(image)
        z_text = self.encode_text([text])

        # Safety scores (higher = safer)
        image_safety = (z_image * self.v_safe).sum(dim=-1).item()
        text_safety = (z_text * self.v_safe).sum(dim=-1).item()

        # Prompt alignment
        alignment = (z_image * z_text).sum(dim=-1).item()

        return {
            "image_safety": image_safety,
            "text_safety": text_safety,
            "alignment": alignment
        }

    def classify_4category(
        self,
        image,
        text: str,
        threshold: float = 0.0
    ) -> Tuple[int, Dict[str, float]]:
        """
        Classify into 4 categories based on image/text safety

        Args:
            image: PIL Image
            text: prompt string
            threshold: safety threshold (default 0.0)

        Returns:
            category (1-4), scores dict

        Categories:
            1: Both safe (image_safety > θ AND text_safety > θ)
            2: Unsafe image, safe text (image_safety ≤ θ AND text_safety > θ)
            3: Safe image, unsafe text (image_safety > θ AND text_safety ≤ θ)
            4: Both unsafe (image_safety ≤ θ AND text_safety ≤ θ)
        """
        scores = self.compute_safety_scores(image, text)

        img_safe = scores["image_safety"] > threshold
        txt_safe = scores["text_safety"] > threshold

        if img_safe and txt_safe:
            category = 1  # Both safe
        elif not img_safe and txt_safe:
            category = 2  # Unsafe image, safe text
        elif img_safe and not txt_safe:
            category = 3  # Safe image, unsafe text
        else:
            category = 4  # Both unsafe

        scores["category"] = category
        scores["threshold"] = threshold

        return category, scores

    def steer_text_toward_safe(self, text: str, alpha: float = 0.5) -> torch.Tensor:
        """
        Steer text embedding toward safe direction (translation-based).

        z_steered = normalize(z + alpha * v_safe)

        Args:
            text: original text prompt
            alpha: steering strength (higher = more steering)

        Returns:
            steered text embedding
        """
        if self.v_safe is None:
            raise ValueError("Must call build_safety_direction() first!")

        # Get original text embedding
        z_text = self.encode_text([text])

        # Steer toward safe direction
        z_steered = z_text + alpha * self.v_safe

        # Renormalize
        z_steered = F.normalize(z_steered, dim=-1)

        return z_steered

    def steer_text_projection(self, text: str, alpha: float = 0.5) -> torch.Tensor:
        """
        Projection-based steering: decompose embedding into content and
        safety components, then only replace the safety component.

        z = z_perp + s * v_safe
          where s = z · v_safe  (safety score)
                z_perp           (content — orthogonal to v_safe)

        z_steered = normalize(z_perp + alpha * v_safe)

        Effect:
        - Unsafe text (s << 0): large unsafe component removed, +alpha added
        - Safe text (s ≈ 0):    barely changes (z_perp ≈ z)
        - Content semantics preserved in z_perp regardless

        Args:
            text: original text prompt
            alpha: safety component magnitude (higher = stronger safe bias)

        Returns:
            steered text embedding (normalized)
        """
        if self.v_safe is None:
            raise ValueError("Must call build_safety_direction() first!")

        # Get original text embedding
        z_text = self.encode_text([text])

        # Decompose: project onto v_safe to get safety component
        s = (z_text * self.v_safe).sum(dim=-1, keepdim=True)  # [1, 1]
        z_perp = z_text - s * self.v_safe  # content (orthogonal to v_safe)

        # Replace safety component with +alpha, keep content intact
        z_steered = z_perp + alpha * self.v_safe

        # Renormalize to unit sphere
        z_steered = F.normalize(z_steered, dim=-1)

        return z_steered

    def compute_reward(
        self,
        image,
        text: str,
        beta_safety: float = 0.5,
        penalize_unsafe: bool = True,
        steer_negative_text: bool = False,
        steering_alpha: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute reward with safety steering

        Args:
            image: PIL Image
            text: prompt string
            beta_safety: weight for safety term
            penalize_unsafe: if True, penalize unsafe content heavily
            steer_negative_text: if True, steer negative text toward safe
            steering_alpha: text steering strength

        Returns:
            dict with reward components
        """
        scores = self.compute_safety_scores(image, text)

        # Check if text is negative
        text_is_negative = scores["text_safety"] < 0

        # Compute base reward (alignment)
        if steer_negative_text and text_is_negative:
            # Steer text toward safe, then compute alignment
            z_image = self.encode_image(image)
            z_text_steered = self.steer_text_toward_safe(text, alpha=steering_alpha)
            base_reward = (z_image * z_text_steered).sum(dim=-1).item()
            steered = True
        else:
            # Normal alignment
            base_reward = scores["alignment"]
            steered = False

        # Safety reward (average of image and text safety)
        safety_reward = (scores["image_safety"] + scores["text_safety"]) / 2.0

        # Combined reward
        if penalize_unsafe and safety_reward < 0:
            # Heavy penalty for unsafe content
            total_reward = base_reward + beta_safety * safety_reward * 2.0
        else:
            total_reward = base_reward + beta_safety * safety_reward

        return {
            "total_reward": total_reward,
            "base_reward": base_reward,
            "safety_reward": safety_reward,
            "text_steered": steered,
            **scores
        }


# ============================================================================
# Threshold Calibration Helper
# ============================================================================

class ThresholdCalibrator:
    """Helper to find optimal threshold from labeled validation data"""

    @staticmethod
    def find_optimal_threshold(
        safety_classifier: SafetySteering,
        validation_data: List[Tuple[Image.Image, str, int]],
        metric: str = "accuracy"
    ) -> Dict:
        """
        Find optimal threshold that maximizes classification metric

        Args:
            safety_classifier: SafetySteering instance
            validation_data: List of (image, text, true_category)
            metric: "accuracy", "f1", or "balanced_accuracy"

        Returns:
            dict with optimal_threshold and metrics
        """
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

        # Compute all safety scores first
        all_scores = []
        true_categories = []

        for image, text, true_cat in validation_data:
            scores = safety_classifier.compute_safety_scores(image, text)
            all_scores.append((scores["image_safety"], scores["text_safety"]))
            true_categories.append(true_cat)

        all_scores = np.array(all_scores)
        true_categories = np.array(true_categories)

        # Try different thresholds
        thresholds = np.linspace(-0.5, 0.5, 50)
        best_threshold = 0.0
        best_score = 0.0
        results = []

        for threshold in thresholds:
            # Classify with this threshold
            predicted = []
            for img_safety, txt_safety in all_scores:
                img_safe = img_safety > threshold
                txt_safe = txt_safety > threshold

                if img_safe and txt_safe:
                    cat = 1
                elif not img_safe and txt_safe:
                    cat = 2
                elif img_safe and not txt_safe:
                    cat = 3
                else:
                    cat = 4
                predicted.append(cat)

            predicted = np.array(predicted)

            # Compute metric
            if metric == "accuracy":
                score = accuracy_score(true_categories, predicted)
            elif metric == "f1":
                score = f1_score(true_categories, predicted, average="weighted")
            elif metric == "balanced_accuracy":
                score = balanced_accuracy_score(true_categories, predicted)

            results.append({
                "threshold": threshold,
                "score": score,
                "accuracy": accuracy_score(true_categories, predicted)
            })

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return {
            "optimal_threshold": best_threshold,
            "best_score": best_score,
            "metric": metric,
            "all_results": results
        }


