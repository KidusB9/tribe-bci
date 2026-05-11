"""
EEG Encoder Network.

Maps raw/processed EEG signals toward TRIBE v2's latent space. The key
insight: 4 EEG channels cannot possibly carry 1152 dimensions of
information. That's like looking through a keyhole and painting the room.

Architecture uses a realistic information bottleneck:
- EEG channels -> small bottleneck (32-64 dim for Muse, 128 for Emotiv)
- Bottleneck is then projected UP to TRIBE's 1152-dim via the domain adapter
- The adapter learns which subset of TRIBE's latent space is actually
  observable from the limited EEG channels (mainly language/attention areas)

Key design decisions:
- Channel-aware bottleneck: dims proportional to actual EEG information
- Spatial attention: learns which EEG channels carry signal vs noise
- Multi-scale temporal convolutions: captures both fast transients and
  slow cortical dynamics relevant to language processing
- Transformer layers: models long-range temporal dependencies
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


def _compute_bottleneck_dim(n_channels: int) -> int:
    """Compute realistic bottleneck dimension based on channel count.

    Information capacity of EEG is roughly proportional to n_channels * bandwidth.
    Muse (4ch): ~32-64 dim is realistic
    Emotiv (14ch): ~128 dim
    OpenBCI Daisy (16ch): ~128-192 dim
    """
    return min(max(n_channels * 16, 32), 256)


@dataclass
class EEGEncoderConfig:
    n_channels: int = 4
    n_samples: int = 512
    hidden_dim: int = 256
    latent_dim: int = 1152  # Final target (TRIBE v2's hidden dimension)
    bottleneck_dim: int = 0  # Auto-computed from n_channels if 0
    n_temporal_scales: int = 4
    n_transformer_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    use_spatial_attention: bool = True
    use_channel_embedding: bool = True

    def __post_init__(self):
        if self.bottleneck_dim == 0:
            self.bottleneck_dim = _compute_bottleneck_dim(self.n_channels)


class SpatialAttention(nn.Module):
    """Learn which EEG channels carry useful neural information.

    Consumer EEGs have channels at very different positions relative to
    speech/language cortex. This module learns to weight them accordingly.
    A Muse headset has TP9/TP10 near auditory cortex (good for speech)
    and AF7/AF8 near prefrontal cortex (good for language semantics).
    """

    def __init__(self, n_channels: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_channels),
            nn.Softmax(dim=-1),
        )
        self.channel_importance = nn.Parameter(torch.ones(n_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # Compute global channel statistics
        channel_stats = torch.cat([
            x.mean(dim=-1),
            x.std(dim=-1),
            x.max(dim=-1).values,
            x.min(dim=-1).values,
        ], dim=-1).reshape(x.shape[0], -1)

        # Project to attention weights
        attn = self.attention(channel_stats[:, :x.shape[1]])
        attn = attn * F.softplus(self.channel_importance)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        return x * attn.unsqueeze(-1), attn


class MultiScaleTemporalConv(nn.Module):
    """Capture neural dynamics at multiple temporal scales.

    Language-related neural activity spans:
    - Fast (10-50ms): phonemic processing, auditory encoding
    - Medium (50-200ms): word recognition, N400 component
    - Slow (200-1000ms): sentence integration, P600 component
    - Very slow (1-2s): discourse-level processing
    """

    def __init__(self, in_channels: int, out_channels: int, n_scales: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        kernel_sizes = [3, 7, 15, 31][:n_scales]

        for ks in kernel_sizes:
            self.convs.append(
                nn.Conv1d(
                    in_channels, out_channels // n_scales,
                    kernel_size=ks, padding=ks // 2, groups=1,
                )
            )
            self.norms.append(nn.LayerNorm(out_channels // n_scales))

        self.fusion = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        outputs = []
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x)  # (B, C_out/n_scales, T)
            h = h.transpose(1, 2)  # (B, T, C_out/n_scales)
            h = norm(h)
            h = F.gelu(h)
            h = h.transpose(1, 2)  # (B, C_out/n_scales, T)
            outputs.append(h)

        # Concatenate across scales
        out = torch.cat(outputs, dim=1)  # (B, C_out, T)
        out = out.transpose(1, 2)  # (B, T, C_out)
        out = self.fusion(out)  # (B, T, C_out)
        return out.transpose(1, 2)  # (B, C_out, T)


class TemporalTransformerBlock(nn.Module):
    """Transformer block for temporal EEG modeling."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


class EEGEncoder(nn.Module):
    """Main EEG encoder that maps raw EEG to TRIBE v2's latent space.

    Architecture:
        Input: (B, n_channels, n_samples)
        -> Channel embedding + Spatial attention
        -> Multi-scale temporal convolutions
        -> Temporal downsampling
        -> Transformer encoder
        -> Global pooling
        -> Projection to TRIBE latent dim (1152)
    """

    def __init__(self, config: Optional[EEGEncoderConfig] = None):
        super().__init__()
        if config is None:
            config = EEGEncoderConfig()
        self.config = config

        # Channel embedding (learnable per-channel features)
        if config.use_channel_embedding:
            self.channel_embed = nn.Parameter(
                torch.randn(1, config.n_channels, config.hidden_dim) * 0.02
            )
            self.channel_proj = nn.Linear(1, config.hidden_dim)
        else:
            self.channel_proj = nn.Linear(config.n_channels, config.hidden_dim)

        # Spatial attention
        if config.use_spatial_attention:
            self.spatial_attn = SpatialAttention(config.n_channels, config.hidden_dim)

        # Multi-scale temporal processing
        self.temporal_conv = MultiScaleTemporalConv(
            in_channels=config.n_channels,
            out_channels=config.hidden_dim,
            n_scales=config.n_temporal_scales,
        )

        # Temporal downsampling (reduce sequence length for transformer)
        self.downsample = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=4, stride=4),
            nn.GELU(),
        )

        # Positional embedding for transformer
        max_len = config.n_samples // 16 + 1
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, config.hidden_dim) * 0.02
        )

        # Transformer encoder
        self.transformer = nn.ModuleList([
            TemporalTransformerBlock(config.hidden_dim, config.n_heads, config.dropout)
            for _ in range(config.n_transformer_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_dim)

        # Information bottleneck: honest about what EEG can carry
        self.bottleneck = nn.Sequential(
            nn.Linear(config.hidden_dim, config.bottleneck_dim),
            nn.LayerNorm(config.bottleneck_dim),
            nn.GELU(),
        )

        # Projection from bottleneck UP to TRIBE latent dimension
        # The domain adapter will refine this further
        self.latent_proj = nn.Sequential(
            nn.Linear(config.bottleneck_dim, config.bottleneck_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.bottleneck_dim * 4, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )

        # Temporal output projection (for sequence-level outputs)
        self.temporal_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.bottleneck_dim),
            nn.GELU(),
            nn.Linear(config.bottleneck_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Raw EEG tensor of shape (B, n_channels, n_samples)
            return_sequence: If True, return per-timestep embeddings
            return_attention: If True, return spatial attention weights

        Returns:
            Dictionary with:
                'latent': (B, latent_dim) global embedding
                'sequence': (B, T, latent_dim) per-timestep (if requested)
                'attention': (B, n_channels) spatial weights (if requested)
        """
        B = x.shape[0]
        outputs = {}

        # Spatial attention
        attn_weights = None
        if hasattr(self, "spatial_attn"):
            x, attn_weights = self.spatial_attn(x)
            if return_attention:
                outputs["attention"] = attn_weights

        # Multi-scale temporal convolution
        h = self.temporal_conv(x)  # (B, hidden, T)

        # Temporal downsampling
        h = self.downsample(h)  # (B, hidden, T//16)

        # Prepare for transformer
        h = h.transpose(1, 2)  # (B, T//16, hidden)
        T = h.shape[1]

        # Add positional embedding
        h = h + self.pos_embed[:, :T, :]

        # Transformer layers
        for block in self.transformer:
            h = block(h)
        h = self.norm(h)  # (B, T//16, hidden)

        # Global pooling -> bottleneck -> project up
        pooled = h.mean(dim=1)  # (B, hidden)
        bottleneck = self.bottleneck(pooled)  # (B, bottleneck_dim)
        latent = self.latent_proj(bottleneck)  # (B, latent_dim)
        outputs["latent"] = latent
        outputs["bottleneck"] = bottleneck  # For inspection

        # Per-timestep embeddings
        if return_sequence:
            seq = self.temporal_proj(h)  # (B, T//16, latent_dim)
            outputs["sequence"] = seq

        return outputs

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: returns just the latent vector."""
        return self.forward(x)["latent"]


class EEGEncoderFromFeatures(nn.Module):
    """Alternative encoder that works from pre-computed EEG features
    (band powers, connectivity, etc.) rather than raw EEG.

    Useful when you want to use the EEGProcessor's feature extraction
    and just need the neural network mapping to TRIBE latent space.
    """

    def __init__(self, feature_dim: int, latent_dim: int = 1152, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
