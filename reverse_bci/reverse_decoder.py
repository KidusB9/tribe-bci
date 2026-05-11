"""
Reverse Decoder: TRIBE Latent Space -> Text Features.

Inverts TRIBE v2's forward model to recover text (language) features
from the shared multimodal latent space. TRIBE v2's forward path is:

    Text features (LLaMA) -> Projector (-> 384-dim) \
    Audio features (Wav2Vec) -> Projector (-> 384-dim) > Cat -> 1152-dim
    Video features (V-JEPA2) -> Projector (-> 384-dim) /
    -> Combiner -> 1152-dim -> Transformer -> 1152-dim
    -> Low-rank head -> 2048-dim -> Subject predictor -> 20484 vertices

Our reverse path:
    1152-dim latent -> Reverse Transformer -> 1152-dim
    -> Modality Separator (extract text component) -> 384-dim
    -> Reverse Projector -> text feature dim
    -> Text Decoder -> vocabulary logits -> words

We DON'T need to perfectly invert the model. We need to recover enough
information about the text modality to decode what the person is thinking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class ReverseDecoderConfig:
    latent_dim: int = 1152
    text_feature_dim: int = 4096 * 6  # LLaMA 3.2-3B, 6 layers concatenated
    text_proj_dim: int = 384  # TRIBE's per-modality projection dim
    n_transformer_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    low_rank_dim: int = 2048
    n_vertices: int = 20484
    use_tribe_weights: bool = True


class ReverseTransformerBlock(nn.Module):
    """Transformer block for the reverse decoding path."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross attention (if context provided)
        if context is not None:
            h = self.norm2(x)
            h, _ = self.cross_attn(h, context, context)
            x = x + h

        # Feed-forward
        h = self.norm3(x)
        h = self.ffn(h)
        x = x + h
        return x


class ModalitySeparator(nn.Module):
    """Learns to extract the text-specific component from the fused latent.

    TRIBE v2 concatenates text (384d) + audio (384d) + video (384d) = 1152d
    before the combiner MLP and transformer. The combiner and transformer
    mix these modalities together. This module learns to "unmix" them,
    extracting the text-relevant signal.

    Approach: learned attention over the latent dimensions, supervised by
    TRIBE v2's known text projector dimensions during training.
    """

    def __init__(self, latent_dim: int, text_dim: int):
        super().__init__()
        self.extract = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, text_dim),
            nn.LayerNorm(text_dim),
        )

        # Modality attention: which parts of the latent correspond to text
        self.modality_gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply modality gate to emphasize text-relevant dimensions
        gate = self.modality_gate(x)
        gated = x * gate
        return self.extract(gated)


class ReverseDecoder(nn.Module):
    """Decodes TRIBE v2 latent representations back to text features.

    This is the core "reverse" component. Given a latent vector from
    TRIBE's space (whether derived from real fMRI or adapted from EEG),
    it recovers the text/language features that would have produced
    that brain activity pattern.

    Can be initialized with TRIBE v2's pretrained weights for the
    inverse mapping (greatly accelerates training).
    """

    def __init__(self, config: Optional[ReverseDecoderConfig] = None):
        super().__init__()
        if config is None:
            config = ReverseDecoderConfig()
        self.config = config

        # Expand latent to sequence for transformer processing
        self.latent_expand = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim * 4),
            nn.GELU(),
            nn.Unflatten(1, (4, config.latent_dim)),
        )

        # Positional embedding for the expanded sequence
        self.pos_embed = nn.Parameter(
            torch.randn(1, 4, config.latent_dim) * 0.02
        )

        # Reverse transformer: process latent into decodable representation
        self.transformer = nn.ModuleList([
            ReverseTransformerBlock(
                config.latent_dim, config.n_heads, config.dropout
            )
            for _ in range(config.n_transformer_layers)
        ])
        self.norm = nn.LayerNorm(config.latent_dim)

        # Modality separator: extract text-specific features
        self.modality_separator = ModalitySeparator(
            config.latent_dim, config.text_proj_dim
        )

        # Reverse projector: map from TRIBE's text projection space
        # back to the original text feature space
        self.reverse_projector = nn.Sequential(
            nn.Linear(config.text_proj_dim, config.text_proj_dim * 4),
            nn.LayerNorm(config.text_proj_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.text_proj_dim * 4, config.text_proj_dim * 8),
            nn.LayerNorm(config.text_proj_dim * 8),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.text_proj_dim * 8, config.text_feature_dim),
        )

        # Optional: fMRI vertex path (for when we have vertex-level input)
        self.from_vertices = nn.Sequential(
            nn.Linear(config.n_vertices, config.low_rank_dim),
            nn.GELU(),
            nn.Linear(config.low_rank_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        latent: torch.Tensor,
        return_intermediate: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Decode latent representation to text features.

        Args:
            latent: (B, latent_dim) or (B, T, latent_dim) from domain adapter
            return_intermediate: if True, return all intermediate representations

        Returns:
            dict with 'text_features' and optionally intermediate values
        """
        outputs = {}

        # Handle both single-vector and sequence inputs
        if latent.ndim == 2:
            # Single vector: expand to short sequence
            expanded = self.latent_expand(latent)  # (B, 4, latent_dim)
            expanded = expanded + self.pos_embed
        else:
            # Already a sequence
            expanded = latent

        # Reverse transformer
        h = expanded
        for block in self.transformer:
            h = block(h)
        h = self.norm(h)

        if return_intermediate:
            outputs["transformer_output"] = h

        # Pool sequence to single vector
        pooled = h.mean(dim=1)  # (B, latent_dim)

        # Extract text-specific component
        text_proj = self.modality_separator(pooled)  # (B, text_proj_dim)
        if return_intermediate:
            outputs["text_projection"] = text_proj

        # Reverse project to text feature space
        text_features = self.reverse_projector(text_proj)  # (B, text_feature_dim)
        outputs["text_features"] = text_features

        return outputs

    def decode_from_vertices(
        self, vertices: torch.Tensor, return_intermediate: bool = False
    ) -> dict[str, torch.Tensor]:
        """Decode directly from fMRI vertex activations.

        Useful for testing the decoder with actual fMRI data
        or TRIBE v2's predicted vertex activations.

        Args:
            vertices: (B, n_vertices) fMRI activation pattern
        """
        latent = self.from_vertices(vertices)  # (B, latent_dim)
        return self.forward(latent, return_intermediate=return_intermediate)

    @torch.no_grad()
    def init_from_tribe(self, tribe_model: nn.Module):
        """Initialize reverse decoder weights from TRIBE v2's forward model.

        Uses the pseudo-inverse of TRIBE's projection layers to initialize
        the reverse projections. This gives a much better starting point
        than random initialization.
        """
        # Extract TRIBE's text projector weights
        if hasattr(tribe_model, "projectors") and "text" in tribe_model.projectors:
            text_proj = tribe_model.projectors["text"]
            for tribe_layer, rev_layer in zip(
                reversed(list(text_proj.children())),
                self.reverse_projector.children(),
            ):
                if isinstance(tribe_layer, nn.Linear) and isinstance(
                    rev_layer, nn.Linear
                ):
                    # Use pseudo-inverse as initialization
                    W = tribe_layer.weight.data
                    W_pinv = torch.linalg.pinv(W)
                    if rev_layer.weight.shape == W_pinv.shape:
                        rev_layer.weight.data.copy_(W_pinv)

        # Extract TRIBE's low-rank head weights
        if hasattr(tribe_model, "low_rank_head"):
            W = tribe_model.low_rank_head.weight.data
            W_pinv = torch.linalg.pinv(W)
            # Initialize from_vertices first layer with pseudo-inverse
            first_linear = None
            for m in self.from_vertices.modules():
                if isinstance(m, nn.Linear):
                    first_linear = m
                    break
            if first_linear is not None and first_linear.weight.shape[1] == W.shape[0]:
                first_linear.weight.data.copy_(W_pinv[:first_linear.weight.shape[0]])

        # Extract TRIBE's predictor weights (average across subjects)
        if hasattr(tribe_model, "predictor"):
            predictor = tribe_model.predictor
            if hasattr(predictor, "weights"):
                W = predictor.weights.data.mean(dim=0)  # Average across subjects
                W_pinv = torch.linalg.pinv(W)
                # Use for the vertex-to-latent path
                for m in self.from_vertices.modules():
                    if isinstance(m, nn.Linear):
                        if m.weight.shape == W_pinv.shape:
                            m.weight.data.copy_(W_pinv)
                        break
