"""Loss functions for generator LoRA fine-tuning.

Three loss components:
1. L_diff: Diffusion denoising loss (preserve diffusion prior)
2. L_identity: Semantic identity loss via DINO/CLIP (ensure same place)
3. L_diverse: Diversity loss (prevent collapse)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DiffusionLoss:
    """Standard diffusion denoising loss.

    Ensures the generator maintains its denoising capability.
    L_diff = MSE(predicted_noise, true_noise)
    """

    def __call__(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE between predicted and true noise.

        Args:
            noise_pred: Predicted noise from UNet (B, C, H, W)
            noise_target: True noise added to latent (B, C, H, W)

        Returns:
            Scalar loss
        """
        return F.mse_loss(noise_pred.float(), noise_target.float())


class IdentityLoss:
    """Semantic identity preservation via pretrained feature extractor.

    Ensures generated image represents the same place as source.
    Uses DINO or CLIP features with cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "dinov2",
        device: str = "cuda",
    ):
        """Initialize identity loss.

        Args:
            model_name: "dinov2" or "clip"
            device: torch device
        """
        self.device = device
        self.model_name = model_name

        if model_name == "dinov2":
            self.model = self._load_dinov2()
            self.preprocess = self._dinov2_preprocess
        elif model_name == "clip":
            self.model, self.processor = self._load_clip()
            self.preprocess = self._clip_preprocess
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _load_dinov2(self):
        """Load DINOv2 ViT-B/14 model."""
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        except Exception as e:
            # Fallback: load from local if available
            raise RuntimeError(f"Failed to load DINOv2: {e}")
        return model.to(self.device)

    def _load_clip(self):
        """Load CLIP ViT-B/32 model."""
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor

    def _dinov2_preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess for DINOv2: images in [0,1] RGB."""
        # DINOv2 expects ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        # Resize to 224x224
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        return (images - mean) / std

    def _clip_preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess for CLIP: images in [0,1] RGB."""
        # CLIP uses similar normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        return (images - mean) / std

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: Batch of images in [0,1] range (B, 3, H, W)

        Returns:
            Normalized feature vectors (B, D)
        """
        preprocessed = self.preprocess(images)

        if self.model_name == "dinov2":
            features = self.model(preprocessed)
        else:  # clip
            features = self.model.get_image_features(pixel_values=preprocessed)

        return F.normalize(features.float(), dim=-1)

    def __call__(
        self,
        generated_images: torch.Tensor,
        source_images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute identity loss.

        Args:
            generated_images: Generated images in [0,1] (B, 3, H, W)
            source_images: Source images in [0,1] (B, 3, H, W)

        Returns:
            Scalar loss (1 - cosine_similarity)
        """
        gen_feat = self.extract_features(generated_images)
        src_feat = self.extract_features(source_images)

        # Cosine similarity loss: minimize 1 - cos(gen, src)
        cosine_sim = (gen_feat * src_feat).sum(dim=-1).mean()
        return 1.0 - cosine_sim


class DiversityLoss:
    """Diversity loss to prevent mode collapse.

    Encourages generated images to be different from each other
    and from previously generated images in the pool.
    """

    def __init__(self, feature_extractor: str = "lpips", device: str = "cuda"):
        """Initialize diversity loss.

        Args:
            feature_extractor: "lpips" or "simple" (pixel-level)
            device: torch device
        """
        self.device = device
        self.feature_extractor = feature_extractor

        if feature_extractor == "lpips":
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net="vgg").to(device)
                self.lpips_model.eval()
                for param in self.lpips_model.parameters():
                    param.requires_grad_(False)
            except ImportError:
                raise RuntimeError(
                    "LPIPS not available. Install with: pip install lpips"
                )

    def _compute_pairwise_diversity(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise diversity within a batch.

        Args:
            images: Batch of images (B, 3, H, W) in [0,1]

        Returns:
            Mean pairwise distance
        """
        if self.feature_extractor == "lpips":
            # LPIPS expects images in [-1, 1]
            images_scaled = images * 2.0 - 1.0
            batch_size = images.shape[0]
            if batch_size < 2:
                return torch.tensor(0.0, device=self.device)

            distances = []
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    dist = self.lpips_model(
                        images_scaled[i:i+1],
                        images_scaled[j:j+1],
                    )
                    distances.append(dist)

            return torch.stack(distances).mean()

        else:  # simple pixel-level
            # Compute pairwise L2 distance in pixel space
            batch_size = images.shape[0]
            if batch_size < 2:
                return torch.tensor(0.0, device=self.device)

            flat = images.flatten(start_dim=1)
            distances = torch.cdist(flat, flat, p=2)
            # Take upper triangle (exclude diagonal)
            mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
            return distances[mask].mean()

    def __call__(
        self,
        generated_images: torch.Tensor,
        pool_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute diversity loss.

        Encourages:
        1. Generated images in this batch to be different from each other
        2. Generated images to be different from pool images (optional)

        Loss is NEGATIVE diversity (we want to maximize diversity = minimize negative)

        Args:
            generated_images: Current batch (B, 3, H, W) in [0,1]
            pool_images: Optional existing pool (P, 3, H, W) in [0,1]

        Returns:
            Scalar loss (negative mean distance)
        """
        # Intra-batch diversity
        intra_diversity = self._compute_pairwise_diversity(generated_images)

        # Pool diversity (if provided)
        if pool_images is not None and pool_images.shape[0] > 0:
            # Compare each generated image to random samples from pool
            n_pool_samples = min(pool_images.shape[0], generated_images.shape[0] * 2)
            indices = torch.randperm(pool_images.shape[0])[:n_pool_samples]
            pool_sample = pool_images[indices]

            combined = torch.cat([generated_images, pool_sample], dim=0)
            combined_diversity = self._compute_pairwise_diversity(combined)

            # Weight both components
            total_diversity = 0.5 * intra_diversity + 0.5 * combined_diversity
        else:
            total_diversity = intra_diversity

        # Return NEGATIVE diversity (loss should be minimized)
        return -total_diversity
