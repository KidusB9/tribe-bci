"""
Domain Adaptation: EEG -> TRIBE v2 Latent Space.

This is the hardest and most critical component. TRIBE v2 was trained on fMRI
(high spatial resolution, low temporal resolution). Consumer EEG is the opposite
(low spatial resolution, high temporal resolution). The domain gap is enormous.

Strategy:
1. Contrastive alignment: For shared stimuli (e.g., same video shown to fMRI
   and EEG subjects), learn to align their latent representations.
2. TRIBE-regularized mapping: Use TRIBE v2's forward model as a teacher.
   The EEG encoder's output should, when passed through TRIBE's decoder,
   produce plausible brain activity patterns.
3. Adversarial domain confusion: A discriminator tries to tell EEG-derived
   latents from fMRI-derived latents; the encoder fools it.
4. Self-supervised pre-training: Use EEG data alone with masked prediction,
   contrastive learning between augmented views, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class DomainAdapterConfig:
    eeg_latent_dim: int = 1152
    tribe_latent_dim: int = 1152
    projection_dim: int = 512
    temperature: float = 0.07
    use_adversarial: bool = True
    use_contrastive: bool = True
    use_reconstruction: bool = True
    adversarial_weight: float = 0.1
    contrastive_weight: float = 1.0
    reconstruction_weight: float = 0.5
    n_tribe_vertices: int = 20484


class ProjectionHead(nn.Module):
    """Projects latent vectors to a shared alignment space."""

    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class DomainDiscriminator(nn.Module):
    """Adversarial discriminator: tries to distinguish EEG-derived
    latents from fMRI-derived (TRIBE) latents.

    Uses gradient reversal during training so the encoder learns to
    produce domain-invariant representations.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.LayerNorm(latent_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GradientReversal(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class LatentRefiner(nn.Module):
    """Refines EEG-derived latent to better match TRIBE's latent distribution.

    Acts as a learned "translator" from the EEG encoder's output space
    to TRIBE v2's expected latent distribution.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.refine(x)
        gate = self.gate(x)
        return x * gate + refined * (1 - gate)


class DomainAdapter(nn.Module):
    """Main domain adaptation module bridging EEG and TRIBE v2 latent spaces.

    Training modes:
    1. Paired mode: EEG + fMRI from same stimuli (best accuracy, needs paired data)
    2. Unpaired mode: EEG and fMRI from different stimuli (uses adversarial alignment)
    3. Self-supervised mode: EEG only (uses TRIBE as teacher via reconstruction loss)
    """

    def __init__(self, config: Optional[DomainAdapterConfig] = None):
        super().__init__()
        if config is None:
            config = DomainAdapterConfig()
        self.config = config

        # Latent refinement (always used)
        self.refiner = LatentRefiner(config.eeg_latent_dim)

        # Contrastive alignment heads
        if config.use_contrastive:
            self.eeg_proj = ProjectionHead(config.eeg_latent_dim, config.projection_dim)
            self.tribe_proj = ProjectionHead(
                config.tribe_latent_dim, config.projection_dim
            )

        # Adversarial domain confusion
        if config.use_adversarial:
            self.discriminator = DomainDiscriminator(config.tribe_latent_dim)
            self.adversarial_lambda = 1.0

        # Reconstruction path: EEG latent -> pseudo-fMRI vertices
        if config.use_reconstruction:
            self.fmri_reconstructor = nn.Sequential(
                nn.Linear(config.tribe_latent_dim, config.tribe_latent_dim * 2),
                nn.GELU(),
                nn.Linear(config.tribe_latent_dim * 2, config.n_tribe_vertices),
            )

    def forward(self, eeg_latent: torch.Tensor) -> torch.Tensor:
        """Map EEG-derived latent to TRIBE-compatible latent.

        Args:
            eeg_latent: (B, eeg_latent_dim) from EEG encoder

        Returns:
            (B, tribe_latent_dim) refined latent in TRIBE's space
        """
        return self.refiner(eeg_latent)

    def compute_contrastive_loss(
        self,
        eeg_latent: torch.Tensor,
        tribe_latent: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss to align EEG and TRIBE latent spaces.

        Requires paired data: same stimulus shown to both EEG and fMRI subjects.
        eeg_latent[i] and tribe_latent[i] correspond to the same stimulus.
        """
        eeg_proj = self.eeg_proj(eeg_latent)  # (B, proj_dim)
        tribe_proj = self.tribe_proj(tribe_latent)  # (B, proj_dim)

        # Cosine similarity matrix
        logits = torch.matmul(eeg_proj, tribe_proj.T) / self.config.temperature
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)

        # Symmetric contrastive loss
        loss_eeg = F.cross_entropy(logits, labels)
        loss_tribe = F.cross_entropy(logits.T, labels)

        return (loss_eeg + loss_tribe) / 2

    def compute_adversarial_loss(
        self,
        eeg_latent: torch.Tensor,
        tribe_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adversarial domain confusion loss.

        Returns:
            discriminator_loss: For training the discriminator
            generator_loss: For training the EEG encoder (via gradient reversal)
        """
        # Refined EEG latent
        refined = self.refiner(eeg_latent)

        # Discriminator predictions
        eeg_pred = self.discriminator(refined)
        tribe_pred = self.discriminator(tribe_latent.detach())

        # Discriminator loss: classify domains correctly
        d_loss_eeg = F.binary_cross_entropy_with_logits(
            eeg_pred, torch.zeros_like(eeg_pred)
        )
        d_loss_tribe = F.binary_cross_entropy_with_logits(
            tribe_pred, torch.ones_like(tribe_pred)
        )
        discriminator_loss = (d_loss_eeg + d_loss_tribe) / 2

        # Generator loss: fool the discriminator (via gradient reversal)
        reversed_eeg = GradientReversal.apply(refined, self.adversarial_lambda)
        gen_pred = self.discriminator(reversed_eeg)
        generator_loss = F.binary_cross_entropy_with_logits(
            gen_pred, torch.ones_like(gen_pred)
        )

        return discriminator_loss, generator_loss

    def compute_reconstruction_loss(
        self,
        eeg_latent: torch.Tensor,
        target_fmri: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruction loss: EEG latent should produce plausible fMRI patterns.

        This uses TRIBE v2's forward model as a teacher signal.
        target_fmri: (B, n_vertices) from TRIBE's predictions for the same stimuli.
        """
        refined = self.refiner(eeg_latent)
        pred_fmri = self.fmri_reconstructor(refined)
        return F.mse_loss(pred_fmri, target_fmri)

    def compute_total_loss(
        self,
        eeg_latent: torch.Tensor,
        tribe_latent: Optional[torch.Tensor] = None,
        target_fmri: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined adaptation loss.

        Works in degraded mode if some signals are unavailable:
        - No tribe_latent: skip contrastive and adversarial losses
        - No target_fmri: skip reconstruction loss
        """
        losses = {}
        total = torch.tensor(0.0, device=eeg_latent.device)

        if tribe_latent is not None:
            if self.config.use_contrastive:
                contrastive = self.compute_contrastive_loss(eeg_latent, tribe_latent)
                losses["contrastive"] = contrastive
                total = total + self.config.contrastive_weight * contrastive

            if self.config.use_adversarial:
                d_loss, g_loss = self.compute_adversarial_loss(eeg_latent, tribe_latent)
                losses["discriminator"] = d_loss
                losses["adversarial"] = g_loss
                total = total + self.config.adversarial_weight * g_loss

        if target_fmri is not None and self.config.use_reconstruction:
            recon = self.compute_reconstruction_loss(eeg_latent, target_fmri)
            losses["reconstruction"] = recon
            total = total + self.config.reconstruction_weight * recon

        losses["total"] = total
        return losses

    def set_adversarial_lambda(self, progress: float):
        """Gradually increase adversarial strength during training.

        progress: float in [0, 1] indicating training progress.
        """
        self.adversarial_lambda = 2.0 / (1.0 + (-10.0 * progress)) - 1.0
