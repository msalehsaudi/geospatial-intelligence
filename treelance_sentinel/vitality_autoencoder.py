"""Pixel-based vitality/morbidity classification using a Conv-Transformer autoencoder.

This computes **pixel-based vitality classes** from two real Sentinel snapshots (previous/current).

Inputs (per tile) come from the time series mode outputs on S3:
- change_detection/tile_<TILE>/time_step_1.tif (current; tree masked; includes NDMI and NDVI bands)
- change_detection/tile_<TILE>/time_step_2.tif (previous; tree masked; includes NDMI and NDVI bands)

We build a per-pixel feature vector and train a Conv-Transformer autoencoder (unsupervised) to model "typical" behavior.
The architecture combines:
- Convolutional layers for local pattern extraction
- Transformer attention for temporal/feature relationships
- Better anomaly detection through hierarchical representations

Then we compute reconstruction error per pixel and convert it to 4 classes:
- 3: high_vitality
- 2: medium_vitality
- 1: medium_morbidity
- 0: severe_morbidity

Important: With only two snapshots, "ground truth" metrics don't exist unless you provide labels.
This module therefore reports *unsupervised* diagnostics (error quantiles, rates per class, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import torch
import torch.nn as nn
from loguru import logger


CLASS_NAMES: dict[int, str] = {
    0: "severe_morbidity",
    1: "medium_morbidity",
    2: "medium_vitality",
    3: "high_vitality",
    4: "no_change",
}


@dataclass(frozen=True)
class ErrorThresholds:
    p50: float
    p75: float
    medium_q: float  # Medium morbidity cutoff, e.g. 85th percentile (for 10% medium morbidity)
    severe_q: float  # High-error cutoff, e.g. 95th percentile (for 5% severe morbidity)


class MlpAutoencoder(nn.Module):
    """Legacy MLP autoencoder (kept for backward compatibility)."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SpatialAttention(nn.Module):
    """
    Spatial Attention Head ("The Eye"): Operates on high-resolution bands (B02, B03, B04, B08).
    Maintains structural integrity and detects canopy thinning, gaps, and structural crown loss.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels, bias=False),
            nn.Sigmoid()
        )
        # Priority weights for spatial bands (B02, B03, B04, B08)
        self.band_bias = nn.Parameter(torch.ones(channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len) or (batch, channels)
        Returns:
            Attention-weighted tensor
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, channels) -> (batch, channels, 1)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y * self.band_bias.view(1, -1, 1)


class SpectralAttention(nn.Module):
    """
    Spectral Attention Head ("The Lab"): Operates across the Channel Dimension (B05, B11, B12).
    Understands spectral coherence and the hidden correlation between Red-Edge (B5) and SWIR (B11).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels, bias=False),
            nn.Sigmoid()
        )
        # Priority weights for spectral bands (B05, B11, B12) - higher bias for physiology
        self.band_bias = nn.Parameter(torch.ones(channels) * 1.5)  # Emphasize spectral bands
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len) or (batch, channels)
        Returns:
            Attention-weighted tensor
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, channels) -> (batch, channels, 1)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y * self.band_bias.view(1, -1, 1)


class ConvTransformerAutoencoder(nn.Module):
    """
    Convolutional-Transformer Autoencoder with Dual-Head Attention for pixel-level vitality detection.
    
    Architecture:
    1. Feature embedding: Projects input features to higher dimension
    2. Dual-Head Attention:
       - Spatial Head ("The Eye"): B02, B03, B04, B08 - structural integrity
       - Spectral Head ("The Lab"): B05, B11, B12 - spectral coherence
    3. Convolutional encoder: Extracts local patterns from feature sequence
    4. Transformer encoder: Captures temporal/sequential relationships via self-attention
    5. Latent bottleneck: Compressed representation
    6. Transformer decoder: Reconstructs with cross-attention
    7. Convolutional decoder: Final reconstruction refinement
    
    This architecture is better suited for:
    - Tri-temporal sequences (T0+T1 input, T2 reconstruction)
    - Capturing relationships between features
    - Learning hierarchical representations
    - Better anomaly detection through attention mechanisms
    """
    def __init__(
        self, 
        input_dim: int,
        latent_dim: int = 8,
        embed_dim: int = 64,
        conv_channels: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.25,
        attention_head_mode: str = "both",
        spatial_band_indices: list[int] | None = None,
        spectral_band_indices: list[int] | None = None,
        target_dim: int | None = None,  # For T0+T1 input, T2 reconstruction
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim if target_dim is not None else input_dim
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.attention_head_mode = attention_head_mode
        self.use_dual_attention = attention_head_mode == "both"
        self.enable_spatial_attention = attention_head_mode in ("both", "spatial")
        self.enable_spectral_attention = attention_head_mode in ("both", "spectral")
        self.spatial_band_indices = spatial_band_indices or []
        self.spectral_band_indices = spectral_band_indices or []
        
        # Input embedding: Project features to higher dimension
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        
        # Dual-Head Attention: Split into Spatial and Spectral branches
        if self.use_dual_attention and len(spatial_band_indices) > 0 and len(spectral_band_indices) > 0:
            # Spatial branch: B02, B03, B04, B08 (structural integrity)
            spatial_input_dim = len(spatial_band_indices)
            self.spatial_embedding = nn.Sequential(
                nn.Linear(spatial_input_dim, embed_dim // 2),
                nn.LayerNorm(embed_dim // 2),
                nn.GELU(),
            )
            self.spatial_attention = SpatialAttention(embed_dim // 2)
            
            # Spectral branch: B05, B11, B12 (spectral coherence)
            spectral_input_dim = len(spectral_band_indices)
            self.spectral_embedding = nn.Sequential(
                nn.Linear(spectral_input_dim, embed_dim // 2),
                nn.LayerNorm(embed_dim // 2),
                nn.GELU(),
            )
            self.spectral_attention = SpectralAttention(embed_dim // 2)
            
            # Combine spatial and spectral branches
            self.branch_combine = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
        else:
            self.spatial_embedding = None
            self.spectral_embedding = None
            self.spatial_attention = None
            self.spectral_attention = None
            self.branch_combine = None

            if self.enable_spatial_attention and len(spatial_band_indices) > 0:
                spatial_input_dim = len(spatial_band_indices)
                self.spatial_embedding = nn.Sequential(
                    nn.Linear(spatial_input_dim, embed_dim // 2),
                    nn.LayerNorm(embed_dim // 2),
                    nn.GELU(),
                )
                self.spatial_attention = SpatialAttention(embed_dim // 2)

            if self.enable_spectral_attention and len(spectral_band_indices) > 0:
                spectral_input_dim = len(spectral_band_indices)
                self.spectral_embedding = nn.Sequential(
                    nn.Linear(spectral_input_dim, embed_dim // 2),
                    nn.LayerNorm(embed_dim // 2),
                    nn.GELU(),
                )
                self.spectral_attention = SpectralAttention(embed_dim // 2)
        
        # Single-branch projection (for spatial-only or spectral-only)
        self.single_branch_projection = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        
        # Convolutional encoder: Extract local patterns
        # Treat features as sequence: (batch, embed_dim) -> (batch, 1, embed_dim) for conv1d
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels * 2),
            nn.GELU(),
        )
        conv_out_dim = conv_channels * 2
        
        # Transformer encoder: Self-attention for feature relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_out_dim,
            nhead=num_heads,
            dim_feedforward=conv_out_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Latent bottleneck
        self.latent_projection = nn.Sequential(
            nn.Linear(conv_out_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
        # Latent expansion
        self.latent_expand = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, conv_out_dim),
        )
        
        # Transformer decoder: Cross-attention for reconstruction
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=conv_out_dim,
            nhead=num_heads,
            dim_feedforward=conv_out_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Convolutional decoder: Refine reconstruction
        self.conv_decoder = nn.Sequential(
            nn.Conv1d(conv_out_dim, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels * 2),
            nn.GELU(),
            nn.Conv1d(conv_channels * 2, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, embed_dim, kernel_size=3, padding=1),
        )
        
        # Output projection: Back to target dimension (T2 bands)
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, self.target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Dual-Head Attention (T0+T1 input, T2 reconstruction).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               For tri-temporal: T0 (prev_) + T1 (mid_) bands concatenated
               Example: (batch, 14) for [prev_B02, mid_B02, prev_B03, mid_B03, ..., prev_B12, mid_B12]
        
        Returns:
            Reconstructed tensor of shape (batch_size, target_dim)
            For tri-temporal: T2 (curr_) bands only
        """
        batch_size = x.shape[0]
        
        # Dual-Head Attention: Split into Spatial and Spectral branches
        if self.use_dual_attention and self.spatial_attention is not None and self.spectral_attention is not None:
            # Extract spatial bands (B02, B03, B04, B08) from input
            spatial_features = x[:, self.spatial_band_indices]  # (batch, spatial_dim)
            # Extract spectral bands (B05, B11, B12) from input
            spectral_features = x[:, self.spectral_band_indices]  # (batch, spectral_dim)
            
            # Embed and apply attention to each branch
            spatial_embed = self.spatial_embedding(spatial_features)  # (batch, embed_dim//2)
            spatial_attended = self.spatial_attention(spatial_embed)  # (batch, embed_dim//2, 1)
            spatial_attended = spatial_attended.squeeze(-1)  # (batch, embed_dim//2)
            
            spectral_embed = self.spectral_embedding(spectral_features)  # (batch, embed_dim//2)
            spectral_attended = self.spectral_attention(spectral_embed)  # (batch, embed_dim//2, 1)
            spectral_attended = spectral_attended.squeeze(-1)  # (batch, embed_dim//2)
            
            # Combine spatial and spectral branches
            combined = torch.cat([spatial_attended, spectral_attended], dim=1)  # (batch, embed_dim)
            x_embed = self.branch_combine(combined)  # (batch, embed_dim)
        else:
            if self.attention_head_mode == "spatial" and self.spatial_embedding is not None:
                spatial_features = x[:, self.spatial_band_indices]
                spatial_embed = self.spatial_embedding(spatial_features)
                if self.spatial_attention is not None:
                    spatial_attended = self.spatial_attention(spatial_embed).squeeze(-1)
                else:
                    spatial_attended = spatial_embed
                x_embed = self.single_branch_projection(spatial_attended)
            elif self.attention_head_mode == "spectral" and self.spectral_embedding is not None:
                spectral_features = x[:, self.spectral_band_indices]
                spectral_embed = self.spectral_embedding(spectral_features)
                if self.spectral_attention is not None:
                    spectral_attended = self.spectral_attention(spectral_embed).squeeze(-1)
                else:
                    spectral_attended = spectral_embed
                x_embed = self.single_branch_projection(spectral_attended)
            else:
                x_embed = self.input_embedding(x)
        
        # Reshape for conv1d: (batch_size, embed_dim) -> (batch_size, 1, embed_dim)
        x_conv = x_embed.unsqueeze(1)
        
        # Convolutional encoder
        # Shape: (batch_size, 1, embed_dim) -> (batch_size, conv_channels*2, embed_dim)
        conv_encoded = self.conv_encoder(x_conv)
        
        # Memory-efficient: Use smaller sequence length for transformer
        # Pool the conv output to reduce sequence length
        # Shape: (batch_size, conv_channels*2, embed_dim) -> (batch_size, conv_channels*2, 1)
        conv_pooled = conv_encoded.mean(dim=2, keepdim=True)  # (batch, channels, 1)
        
        # Permute for transformer: (batch, channels, seq_len) -> (batch, seq_len, channels)
        # Shape: (batch_size, conv_channels*2, 1) -> (batch_size, 1, conv_channels*2)
        conv_encoded_seq = conv_pooled.permute(0, 2, 1)
        
        # Transformer encoder: Self-attention on single token
        # Shape: (batch_size, 1, conv_channels*2)
        transformer_encoded = self.transformer_encoder(conv_encoded_seq)
        
        # Remove sequence dimension
        # Shape: (batch_size, 1, conv_channels*2) -> (batch_size, conv_channels*2)
        pooled = transformer_encoded.squeeze(1)
        
        # Latent bottleneck
        # Shape: (batch_size, conv_channels*2) -> (batch_size, latent_dim)
        latent = self.latent_projection(pooled)
        
        # Expand latent
        # Shape: (batch_size, latent_dim) -> (batch_size, conv_channels*2)
        latent_expanded = self.latent_expand(latent)
        
        # Reshape for transformer decoder: (batch, features) -> (batch, seq_len, features)
        # Shape: (batch_size, conv_channels*2) -> (batch_size, 1, conv_channels*2)
        latent_seq = latent_expanded.unsqueeze(1)
        
        # Transformer decoder: Cross-attention with encoder output
        # Shape: (batch_size, 1, conv_channels*2)
        transformer_decoded = self.transformer_decoder(latent_seq, transformer_encoded)
        
        # Remove sequence dimension
        # Shape: (batch_size, 1, conv_channels*2) -> (batch_size, conv_channels*2)
        transformer_decoded = transformer_decoded.squeeze(1)
        
        # Reshape for conv1d: (batch, features) -> (batch, channels, seq_len)
        # Shape: (batch_size, conv_channels*2) -> (batch_size, conv_channels*2, 1)
        transformer_decoded_conv = transformer_decoded.unsqueeze(-1)
        
        # Convolutional decoder
        # Shape: (batch_size, conv_channels*2, 1) -> (batch_size, embed_dim, 1)
        conv_decoded = self.conv_decoder(transformer_decoded_conv)
        
        # Remove sequence dimension
        # Shape: (batch_size, embed_dim, 1) -> (batch_size, embed_dim)
        conv_decoded = conv_decoded.squeeze(-1)
        
        # Output projection: Back to target dimension (T2 bands)
        # Shape: (batch_size, embed_dim) -> (batch_size, target_dim)
        output = self.output_projection(conv_decoded)
        
        return output


def _join(base: str, *parts: str) -> str:
    return os.path.join(base, *parts)


def _band_index(src: rasterio.io.DatasetReader, name: str) -> int:
    if not src.descriptions:
        raise ValueError("Raster has no band descriptions; cannot find NDMI/NDVI")
    for i, desc in enumerate(src.descriptions, start=1):
        if (desc or "").strip().upper() == name.upper():
            return i
    raise ValueError(f"Band '{name}' not found; descriptions={src.descriptions}")


def _scaled_u16_to_float(arr: np.ndarray, nodata: int) -> tuple[np.ndarray, np.ndarray]:
    """0..2000 => [-1,1] for NDVI/NDMI scaling used by pipeline."""
    arr_i = arr.astype(np.int32, copy=False)
    valid = arr_i != int(nodata)
    val = (arr_i.astype(np.float32) / 1000.0) - 1.0
    return val, valid


def _band_to_float(arr: np.ndarray, nodata: int, band_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert band to float, handling both scaled indices (NDVI/NDMI) and raw reflectance bands.
    
    - NDVI/NDMI: Scaled 0-2000 -> [-1, 1] (divide by 1000, subtract 1)
    - Raw bands (B02-B12): Typically 0-10000 reflectance -> normalize to [0, 1] (divide by 10000)
    """
    arr_i = arr.astype(np.int32, copy=False)
    valid = arr_i != int(nodata)
    
    # Check if this is a scaled index (NDVI/NDMI) or raw band
    if band_name.upper() in ["NDVI", "NDMI", "EVI", "NDBI", "NDRE"]:
        # Scaled index: 0-2000 -> [-1, 1]
        val = (arr_i.astype(np.float32) / 1000.0) - 1.0
    else:
        # Raw reflectance band: 0-10000 -> [0, 1] (normalize)
        # Sentinel-2 L2A reflectance is typically in 0-10000 range
        val = arr_i.astype(np.float32) / 10000.0
    
    return val, valid


def _thresholds(errors: np.ndarray, severe_percentile: float = 0.95, medium_percentile: float = 0.85) -> ErrorThresholds:
    """
    Compute error thresholds for vitality classes.

    We intentionally use percentile-based cutoffs for morbidity classes
    instead of the classical Tukey fence. This guarantees that roughly
    the top X% of pixels by reconstruction error are assigned to morbidity
    classes, which is more intuitive for change detection maps.

    Args:
        errors: Array of reconstruction errors
        severe_percentile: Percentile threshold for severe morbidity (default: 0.95, i.e., top 5%)
        medium_percentile: Percentile threshold for medium morbidity (default: 0.85, i.e., 85th-95th = 10%)

    Returns:
        ErrorThresholds with:
        - p50: 50th percentile
        - p75: 75th percentile
        - medium_q: medium_percentile-th percentile (e.g., 85th = 10% medium morbidity between 85th-95th)
        - severe_q: severe_percentile-th percentile (e.g., 95th = top 5% highest errors)
    """
    p50 = float(np.quantile(errors, 0.50))
    p75 = float(np.quantile(errors, 0.75))
    medium_q = float(np.quantile(errors, medium_percentile))
    severe_q = float(np.quantile(errors, severe_percentile))
    return ErrorThresholds(
        p50=p50,
        p75=p75,
        medium_q=medium_q,
        severe_q=severe_q,
    )


def _train(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """
    Train autoencoder with memory-efficient batch processing.
    
    For tri-temporal: x_train = T0+T1 input, y_train = T2 target (reconstruction target).
    For Conv-Transformer models, uses smaller effective batch sizes to avoid CUDA OOM errors.
    """
    model.to(device)
    model.train()
    
    # For Conv-Transformer, use smaller effective batch size to avoid OOM
    is_conv_transformer = isinstance(model, ConvTransformerAutoencoder)
    effective_batch_size = min(batch_size, 8192) if is_conv_transformer else batch_size
    
    if is_conv_transformer and batch_size > effective_batch_size:
        logger.info(f"Using memory-efficient batch size: {effective_batch_size} (requested: {batch_size})")
    
    # Verify input and target have same number of samples
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Input/target sample count mismatch: x_train.shape[0]={x_train.shape[0]}, "
            f"y_train.shape[0]={y_train.shape[0]}"
        )
    
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(x_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32)
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=effective_batch_size, shuffle=True, drop_last=False)
    # Add mild L2 regularization to increase sensitivity to deviations
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    for ep in range(1, epochs + 1):
        tot = 0.0
        n = 0
        for xb, yb_target in dl:
            xb = xb.to(device)
            yb_target = yb_target.to(device)
            opt.zero_grad(set_to_none=True)
            yb_pred = model(xb)  # Model predicts T2 from T0+T1 input
            loss = loss_fn(yb_pred, yb_target)  # Compare prediction to T2 target
            loss.backward()
            opt.step()
            tot += float(loss.detach().cpu().item()) * xb.shape[0]
            n += xb.shape[0]
        avg = tot / max(1, n)
        losses.append(avg)
        if ep in {1, epochs} or ep % max(1, epochs // 5) == 0:
            logger.info(f"Epoch {ep:>3}/{epochs} - train_mse={avg:.6f}")
    return losses


def _recon_error(
    model: nn.Module,
    x: np.ndarray,
    y_true: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Compute reconstruction error with memory-efficient batch processing.
    
    For tri-temporal: x = T0+T1 input, y_true = T2 target.
    Error is computed as MSE between predicted T2 and actual T2.
    
    For Conv-Transformer models, uses smaller effective batch sizes to avoid CUDA OOM errors.
    """
    model.eval()
    
    # For Conv-Transformer, use smaller effective batch size to avoid OOM
    is_conv_transformer = isinstance(model, ConvTransformerAutoencoder)
    effective_batch_size = min(batch_size, 8192) if is_conv_transformer else batch_size
    
    # Verify input and target have same number of samples
    if x.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Input/target sample count mismatch: x.shape[0]={x.shape[0]}, "
            f"y_true.shape[0]={y_true.shape[0]}"
        )
    
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], effective_batch_size):
            xb = torch.as_tensor(x[start : start + effective_batch_size], dtype=torch.float32, device=device)
            yb_true = torch.as_tensor(y_true[start : start + effective_batch_size], dtype=torch.float32, device=device)
            yb_pred = model(xb)  # Model predicts T2 from T0+T1 input
            
            # Verify output dimension matches target
            expected_output_dim = model.target_dim if hasattr(model, 'target_dim') else model.input_dim
            if yb_pred.shape[1] != expected_output_dim:
                raise ValueError(
                    f"Autoencoder output dimension mismatch!\n"
                    f"  Input shape: {xb.shape} (features: {xb.shape[1]})\n"
                    f"  Output shape: {yb_pred.shape} (features: {yb_pred.shape[1]})\n"
                    f"  Target shape: {yb_true.shape} (features: {yb_true.shape[1]})\n"
                    f"  Model target_dim: {expected_output_dim}\n"
                    f"  Expected: Model must output {expected_output_dim} features (T2 bands)."
                )
            
            if yb_pred.shape != yb_true.shape:
                raise ValueError(
                    f"Prediction/target shape mismatch: yb_pred.shape={yb_pred.shape}, "
                    f"yb_true.shape={yb_true.shape}"
                )
            
            # Compute MSE between predicted T2 and actual T2
            eb = torch.mean((yb_pred - yb_true) ** 2, dim=1)
            out.append(eb.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def run_tile(
    *,
    base_output_dir: str,
    tile_id: str,
    output_s3_dir: str | None = None,
    epochs: int | None = None,  # None = auto-detect: 14 for Conv-Transformer, 60 for MLP
    lr: float = 1e-3,
    batch_size: int = 262144,
    max_train_pixels: int = 300_000,
    seed: int = 42,
    no_change_abs_delta_ndvi: float | None = 0.03,  # Default: 0.03 (3% change threshold)
    no_change_abs_delta_ndmi: float | None = 0.03,  # Default: 0.03 (3% change threshold)
    classified_raster_s3: str | None = None,
    severe_morbidity_percentile: float = 0.95,
    medium_morbidity_percentile: float = 0.85,
    feature_bands: list[str] | None = None,
    use_directional_classification: bool = False,
    directional_ndvi_threshold: float = 0.05,
    directional_ndmi_threshold: float = 0.05,
    no_change_percentile: float | None = 0.70,  # Target: 70% no-change pixels
    attention_head_mode: str = "both",
) -> dict[str, str]:
    """Train AE on a sample of tree pixels and classify all tree pixels into 4 vitality classes.
    
    If classified_raster_s3 is provided, trains a separate autoencoder per cluster for better
    outlier sensitivity within each cluster.
    
    Default epochs: 8 per cluster (if epochs=None).
    """
    
    # Auto-detect default epochs based on model type (Conv-Transformer uses 8, MLP uses 60)
    if epochs is None:
        epochs = 8  # Default: 8 epochs per cluster
        logger.info(f"Using default epochs: {epochs} (Conv-Transformer autoencoder)")

    if isinstance(base_output_dir, str) and base_output_dir.startswith("s3://"):
        raise ValueError("S3 base_output_dir is no longer supported. Please use a local output directory.")
    if output_s3_dir is None:
        output_s3_dir = _join(base_output_dir, "vitality_autoencoder", f"tile_{tile_id}")
    elif isinstance(output_s3_dir, str) and output_s3_dir.startswith("s3://"):
        raise ValueError("S3 output directories are no longer supported. Please use a local output directory.")

    # Tri-temporal mode: Check for all 3 time steps
    ts0 = _join(base_output_dir, "change_detection", f"tile_{tile_id}", "time_step_0.tif")
    ts1 = _join(base_output_dir, "change_detection", f"tile_{tile_id}", "time_step_1.tif")
    ts2 = _join(base_output_dir, "change_detection", f"tile_{tile_id}", "time_step_2.tif")
    
    # Check if tri-temporal mode is available (time_step_0 exists)
    has_tri_temporal = os.path.exists(ts0)
    if not (os.path.exists(ts1) and os.path.exists(ts2)):
        raise ValueError("Local tri-temporal inputs are required: time_step_1.tif and time_step_2.tif were not found.")
    
    if has_tri_temporal:
        logger.info("Tri-temporal mode detected: time_step_0.tif is available")
    else:
        raise ValueError("Tri-temporal mode required: time_step_0.tif not found. All 3 time steps must be available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    rng = np.random.default_rng(seed)

    ts0_local = ts0
    ts1_local = ts1
    ts2_local = ts2
        
    cluster_raster_local = None
    if classified_raster_s3:
        if isinstance(classified_raster_s3, str) and classified_raster_s3.startswith("s3://"):
            raise ValueError(f"S3 classified raster paths are no longer supported: {classified_raster_s3}")
        cluster_raster_local = classified_raster_s3
        logger.info(f"Loaded classified raster from {classified_raster_s3} for per-cluster processing")

    # Default feature bands if not specified.
    # For tri-temporal: T0 (prev_), T1 (mid_), T2 (curr_)
    if feature_bands is None:
        # Base band names (no prefix) – easier to reason about.
        feature_bands = ["B02", "B03", "B04", "B05", "B08", "B11", "B12"]

    # Normalize feature band specs for tri-temporal:
    # - If the user passed bare names like "B08", "B11", we expand them to
    #   all three time steps: ["prev_B08", "mid_B08", "curr_B08", ...].
    # - If they passed explicit "prev_*" / "mid_*" / "curr_*" names, we keep them.
    normalized_bands: list[str] = []
    for b in feature_bands:
        if b.startswith("prev_") or b.startswith("mid_") or b.startswith("curr_"):
            normalized_bands.append(b)
        else:
            # Tri-temporal: expand to all 3 time steps
            normalized_bands.append(f"prev_{b}")  # T0 (oldest)
            normalized_bands.append(f"mid_{b}")   # T1 (middle)
            normalized_bands.append(f"curr_{b}")  # T2 (current)

    feature_bands = normalized_bands

    logger.info(f"Using feature bands for autoencoder (tri-temporal expanded): {feature_bands}")

    with rasterio.open(ts0_local) as s0, rasterio.open(ts1_local) as s1, rasterio.open(ts2_local) as s2:
        # Tri-temporal: s0=time_step_0 (oldest/T0), s1=time_step_1 (middle/T1), s2=time_step_2 (current/T2)
        logger.info("Running in tri-temporal mode: using time_step_0 (oldest), time_step_1 (middle), time_step_2 (current)")
        
        # Build band index mapping for all three time steps
        band_data = {}  # Will store all needed bands
        nodata0 = int(s0.nodata) if s0.nodata is not None else 0
        nodata1 = int(s1.nodata) if s1.nodata is not None else 0
        nodata2 = int(s2.nodata) if s2.nodata is not None else 0
        
        # Helper function to get band index
        def get_band_idx(src, band_name):
            try:
                return _band_index(src, band_name)
            except ValueError:
                # Try to find by description
                if src.descriptions:
                    for i, desc in enumerate(src.descriptions, start=1):
                        if desc and band_name.upper() in desc.upper():
                            return i
                raise ValueError(f"Band '{band_name}' not found in raster")
        
        # Load bands separately for each time step (prev_/mid_/curr_)
        for band_spec in feature_bands:
            if band_spec.startswith("prev_"):
                band_name = band_spec.replace("prev_", "")
                try:
                    idx = get_band_idx(s0, band_name)  # T0 (oldest)
                    band_array = s0.read(idx)
                    if band_array.shape != (s0.height, s0.width):
                        if band_array.ndim == 3 and band_array.shape[0] == 1:
                            band_array = band_array[0]
                        if band_array.shape != (s0.height, s0.width):
                            raise ValueError(
                                f"Band {band_name} from time_step_0 has shape {band_array.shape}, "
                                f"expected {(s0.height, s0.width)}"
                            )
                    band_data[f"prev_{band_name}"] = band_array
                    logger.debug(f"Loaded prev_{band_name}: shape {band_array.shape}")
                except ValueError as e:
                    logger.warning(f"Could not load previous band {band_name}: {e}. Skipping '{band_spec}'.")
            elif band_spec.startswith("mid_"):
                band_name = band_spec.replace("mid_", "")
                try:
                    idx = get_band_idx(s1, band_name)  # T1 (middle)
                    band_array = s1.read(idx)
                    if band_array.shape != (s1.height, s1.width):
                        if band_array.ndim == 3 and band_array.shape[0] == 1:
                            band_array = band_array[0]
                        if band_array.shape != (s1.height, s1.width):
                            raise ValueError(
                                f"Band {band_name} from time_step_1 (middle) has shape {band_array.shape}, "
                                f"expected {(s1.height, s1.width)}"
                            )
                    band_data[f"mid_{band_name}"] = band_array
                    logger.debug(f"Loaded mid_{band_name}: shape {band_array.shape}")
                except ValueError as e:
                    logger.warning(f"Could not load middle band {band_name}: {e}. Skipping '{band_spec}'.")
            elif band_spec.startswith("curr_"):
                band_name = band_spec.replace("curr_", "")
                try:
                    idx = get_band_idx(s2, band_name)  # T2 (current)
                    band_array = s2.read(idx)
                    if band_array.shape != (s2.height, s2.width):
                        if band_array.ndim == 3 and band_array.shape[0] == 1:
                            band_array = band_array[0]
                        if band_array.shape != (s2.height, s2.width):
                            raise ValueError(
                                f"Band {band_name} from time_step_2 has shape {band_array.shape}, "
                                f"expected {(s2.height, s2.width)}"
                            )
                    band_data[f"curr_{band_name}"] = band_array
                    logger.debug(f"Loaded curr_{band_name}: shape {band_array.shape}")
                except ValueError as e:
                    logger.warning(f"Could not load current band {band_name}: {e}. Skipping '{band_spec}'.")
            else:
                # Legacy: If no prefix, try to load for all three time steps
                band_name = band_spec
                try:
                    idx0 = get_band_idx(s0, band_name)
                    idx1 = get_band_idx(s1, band_name)
                    idx2 = get_band_idx(s2, band_name)
                    band0 = s0.read(idx0)
                    band1 = s1.read(idx1)
                    band2 = s2.read(idx2)
                    # Handle single-band dimension
                    if band0.ndim == 3 and band0.shape[0] == 1:
                        band0 = band0[0]
                    if band1.ndim == 3 and band1.shape[0] == 1:
                        band1 = band1[0]
                    if band2.ndim == 3 and band2.shape[0] == 1:
                        band2 = band2[0]
                    band_data[f"prev_{band_name}"] = band0
                    band_data[f"mid_{band_name}"] = band1
                    band_data[f"curr_{band_name}"] = band2
                except ValueError as e:
                    logger.warning(f"Could not load band {band_name} for all time steps: {e}. Skipping.")
        
        # Load NDVI/NDMI for no_change mask from all time steps
        try:
            ndmi0_i = _band_index(s0, "NDMI")
            ndvi0_i = _band_index(s0, "NDVI")
            ndmi1_i = _band_index(s1, "NDMI")
            ndvi1_i = _band_index(s1, "NDVI")
            ndmi2_i = _band_index(s2, "NDMI")
            ndvi2_i = _band_index(s2, "NDVI")
            ndmi0_u16 = s0.read(ndmi0_i)
            ndvi0_u16 = s0.read(ndvi0_i)
            ndmi1_u16 = s1.read(ndmi1_i)
            ndvi1_u16 = s1.read(ndvi1_i)
            ndmi2_u16 = s2.read(ndmi2_i)
            ndvi2_u16 = s2.read(ndvi2_i)
        except ValueError:
            # If NDVI/NDMI not found, we'll skip no_change mask
            ndmi0_u16 = None
            ndvi0_u16 = None
            ndmi1_u16 = None
            ndvi1_u16 = None
            ndmi2_u16 = None
            ndvi2_u16 = None

        # Ensure all three time steps are pixel-aligned (use T2 as reference)
        needs_reprojection_0 = (
            s0.width != s2.width
            or s0.height != s2.height
            or s0.transform != s2.transform
            or s0.crs != s2.crs
        )
        needs_reprojection_1 = (
            s1.width != s2.width
            or s1.height != s2.height
            or s1.transform != s2.transform
            or s1.crs != s2.crs
        )
        needs_reprojection = needs_reprojection_0 or needs_reprojection_1
        
        # Use T2 (current) as reference grid for all reprojections
        reference_raster = s2
        dst_shape = (s2.height, s2.width)
        
        if needs_reprojection:
            logger.warning(
                f"Time steps are not aligned; reprojecting onto time_step_2 (current) grid. "
                f"Reference shape: {dst_shape}"
            )
            if needs_reprojection_0:
                logger.warning(
                    f"Dimension mismatch: time_step_0 (oldest)={s0.width}x{s0.height}, "
                    f"time_step_2 (current)={s2.width}x{s2.height}. Reprojecting."
                )
            if needs_reprojection_1:
                logger.warning(
                    f"Dimension mismatch: time_step_1 (middle)={s1.width}x{s1.height}, "
                    f"time_step_2 (current)={s2.width}x{s2.height}. Reprojecting."
                )
            
            # Reproject all bands to match T2 (current) dimensions
            for band_key in list(band_data.keys()):
                original = band_data[band_key]
                
                # Determine source raster based on prefix
                if band_key.startswith("prev_"):
                    src_raster = s0
                    src_shape = (s0.height, s0.width)
                    src_nodata = nodata0
                    needs_reproj = needs_reprojection_0
                elif band_key.startswith("mid_"):
                    src_raster = s1
                    src_shape = (s1.height, s1.width)
                    src_nodata = nodata1
                    needs_reproj = needs_reprojection_1
                else:  # curr_
                    src_raster = s2
                    src_shape = (s2.height, s2.width)
                    src_nodata = nodata2
                    needs_reproj = False
                
                if needs_reproj:
                    # Ensure original has correct shape for reprojection
                    if original.shape != src_shape:
                        if original.ndim == 3 and original.shape[0] == 1:
                            original = original[0]
                        if original.size == src_shape[0] * src_shape[1]:
                            original = original.reshape(src_shape)
                        else:
                            raise ValueError(
                                f"Cannot reshape band {band_key}: shape {original.shape}, "
                                f"size {original.size} != {src_shape[0] * src_shape[1]}"
                            )
                    
                    reproj = np.full(dst_shape, src_nodata, dtype=original.dtype)
                    reproject(
                        source=original,
                        destination=reproj,
                        src_transform=src_raster.transform,
                        src_crs=src_raster.crs,
                        dst_transform=s2.transform,
                        dst_crs=s2.crs,
                        resampling=Resampling.nearest,
                    )
                    band_data[band_key] = reproj
                    logger.debug(f"Reprojected {band_key}: {original.shape} → {reproj.shape}")
            
            # Reproject NDVI/NDMI from T0 and T1 if they exist and need reprojection
            if ndmi0_u16 is not None and ndvi0_u16 is not None and needs_reprojection_0:
                for arr_name, arr, arr_nodata in [("ndmi0", ndmi0_u16, nodata0), ("ndvi0", ndvi0_u16, nodata0)]:
                    if arr.shape != (s0.height, s0.width):
                        if arr.ndim == 3 and arr.shape[0] == 1:
                            arr = arr[0]
                        if arr.size == s0.height * s0.width:
                            arr = arr.reshape((s0.height, s0.width))
                    
                    reproj = np.full(dst_shape, arr_nodata, dtype=arr.dtype)
                    reproject(
                        source=arr,
                        destination=reproj,
                        src_transform=s0.transform,
                        src_crs=s0.crs,
                        dst_transform=s2.transform,
                        dst_crs=s2.crs,
                        resampling=Resampling.nearest,
                    )
                    if arr_name == "ndmi0":
                        ndmi0_u16 = reproj
                    else:
                        ndvi0_u16 = reproj
            
            if ndmi1_u16 is not None and ndvi1_u16 is not None and needs_reprojection_1:
                for arr_name, arr, arr_nodata in [("ndmi1", ndmi1_u16, nodata1), ("ndvi1", ndvi1_u16, nodata1)]:
                    if arr.shape != (s1.height, s1.width):
                        if arr.ndim == 3 and arr.shape[0] == 1:
                            arr = arr[0]
                        if arr.size == s1.height * s1.width:
                            arr = arr.reshape((s1.height, s1.width))
                    
                    reproj = np.full(dst_shape, arr_nodata, dtype=arr.dtype)
                    reproject(
                        source=arr,
                        destination=reproj,
                        src_transform=s1.transform,
                        src_crs=s1.crs,
                        dst_transform=s2.transform,
                        dst_crs=s2.crs,
                        resampling=Resampling.nearest,
                    )
                    if arr_name == "ndmi1":
                        ndmi1_u16 = reproj
                    else:
                        ndvi1_u16 = reproj

            # Convert bands to float and build feature vectors
            band_arrays = {}
            valid_masks = {}
            
            # Ensure all bands have the same shape (should be s2.height, s2.width after reprojection)
            expected_shape = (s2.height, s2.width)
            
            # Process all bands
            for band_key, band_u16 in band_data.items():
                # Verify band shape matches expected
                if band_u16.shape != expected_shape:
                    logger.warning(
                        f"Band {band_key} has shape {band_u16.shape}, expected {expected_shape}. "
                        f"Reshaping if possible."
                    )
                    if band_u16.size == expected_shape[0] * expected_shape[1]:
                        band_u16 = band_u16.reshape(expected_shape)
                    else:
                        raise ValueError(
                            f"Cannot reshape band {band_key}: size {band_u16.size} != "
                            f"{expected_shape[0] * expected_shape[1]}"
                        )
                
                # Determine nodata and band name based on prefix
                if band_key.startswith("prev_"):
                    nodata = nodata0
                    band_name = band_key.replace("prev_", "")
                elif band_key.startswith("mid_"):
                    nodata = nodata1
                    band_name = band_key.replace("mid_", "")
                else:  # curr_
                    nodata = nodata2
                    band_name = band_key.replace("curr_", "")
                
                # Use appropriate scaling based on band type
                band_float, valid_mask = _band_to_float(band_u16, nodata, band_name)
                
                # Verify valid_mask has correct shape
                if valid_mask.shape != expected_shape:
                    raise ValueError(
                        f"Valid mask for {band_key} has shape {valid_mask.shape}, "
                        f"expected {expected_shape}"
                    )
                
                band_arrays[band_key] = band_float
                valid_masks[band_key] = valid_mask
            
            # Process NDVI/NDMI for no_change mask if available (use T0 and T2 for comparison)
            if ndmi0_u16 is not None and ndvi0_u16 is not None and ndmi2_u16 is not None and ndvi2_u16 is not None:
                # Ensure NDVI/NDMI arrays have the same shape (should match time_step_2 after reprojection)
                expected_shape = (s2.height, s2.width)
                
                # Verify and fix shapes if needed
                for arr_name, arr in [("ndvi0", ndvi0_u16), ("ndvi2", ndvi2_u16), ("ndmi0", ndmi0_u16), ("ndmi2", ndmi2_u16)]:
                    if arr.shape != expected_shape:
                        if arr.size == expected_shape[0] * expected_shape[1]:
                            arr = arr.reshape(expected_shape)
                        else:
                            raise ValueError(f"{arr_name} shape {arr.shape} cannot be reshaped to {expected_shape}")
                
                ndmi0, v0a = _scaled_u16_to_float(ndmi0_u16, nodata0)
                ndvi0, v0b = _scaled_u16_to_float(ndvi0_u16, nodata0)
                ndmi2, v2a = _scaled_u16_to_float(ndmi2_u16, nodata2)
                ndvi2, v2b = _scaled_u16_to_float(ndvi2_u16, nodata2)
                
                # Verify all NDVI/NDMI arrays have the same shape after processing
                if ndvi0.shape != expected_shape or ndvi2.shape != expected_shape:
                    raise ValueError(
                        f"NDVI shape mismatch after processing: "
                        f"ndvi0.shape={ndvi0.shape}, ndvi2.shape={ndvi2.shape}, expected={expected_shape}"
                    )
                if ndmi0.shape != expected_shape or ndmi2.shape != expected_shape:
                    raise ValueError(
                        f"NDMI shape mismatch after processing: "
                        f"ndmi0.shape={ndmi0.shape}, ndmi2.shape={ndmi2.shape}, expected={expected_shape}"
                    )
            else:
                # Use first band's valid mask as fallback
                first_valid = list(valid_masks.values())[0] if valid_masks else None
                v0a = v0b = v2a = v2b = first_valid if first_valid is not None else np.ones((s2.height, s2.width), dtype=bool)
                ndmi0 = ndvi0 = ndmi2 = ndvi2 = None

            # Combine all valid masks
            all_valid = np.ones((s2.height, s2.width), dtype=bool)
            for valid_mask in valid_masks.values():
                all_valid &= valid_mask
            # Also include NDVI/NDMI valid masks if they exist
            if ndmi0_u16 is not None and ndvi0_u16 is not None and ndmi2_u16 is not None and ndvi2_u16 is not None:
                all_valid &= v0a & v0b & v2a & v2b
            
            valid = all_valid
            n_valid = int(valid.sum())
            if n_valid == 0:
                raise ValueError("No valid (tree) pixels found in the three time steps.")
            logger.info(f"Valid tree pixels: {n_valid:,}")
            
            # Log band shapes for debugging
            logger.debug(f"Band shapes after processing:")
            for band_key, band_array in band_arrays.items():
                logger.debug(f"  {band_key}: {band_array.shape}, valid pixels: {int(valid_masks[band_key].sum()):,}")

            # Define "no_change" pixels using small absolute deltas between T0 and T2
            # Default thresholds: 0.03 (3% change) - pixels with smaller changes are classified as "no_change"
            no_change_mask = valid.copy()
            if no_change_abs_delta_ndvi is not None and ndvi0 is not None and ndvi2 is not None:
                no_change_mask &= np.abs(ndvi0 - ndvi2) <= float(no_change_abs_delta_ndvi)
                logger.info(f"No-change NDVI threshold: {no_change_abs_delta_ndvi} (pixels with |ΔNDVI| <= {no_change_abs_delta_ndvi})")
            if no_change_abs_delta_ndmi is not None and ndmi0 is not None and ndmi2 is not None:
                no_change_mask &= np.abs(ndmi0 - ndmi2) <= float(no_change_abs_delta_ndmi)
                logger.info(f"No-change NDMI threshold: {no_change_abs_delta_ndmi} (pixels with |ΔNDMI| <= {no_change_abs_delta_ndmi})")
            
            n_no_change = int(no_change_mask.sum())
            if n_no_change > 0:
                logger.info(f"No-change pixels: {n_no_change:,} ({100*n_no_change/valid.sum():.2f}% of valid pixels)")
            else:
                logger.warning(f"No pixels classified as 'no_change' (thresholds might be too strict)")

            # Build feature vectors for tri-temporal reconstruction:
            # - Input X: concatenate T0 (prev_) and T1 (mid_) bands
            # - Target Y: T2 (curr_) bands only
            input_bands: list[str] = []
            target_bands: list[str] = []
            for band_spec in feature_bands:
                if band_spec in band_arrays:
                    if band_spec.startswith("prev_") or band_spec.startswith("mid_"):
                        input_bands.append(band_spec)
                    elif band_spec.startswith("curr_"):
                        target_bands.append(band_spec)
                else:
                    logger.warning(
                        f"Band specification '{band_spec}' not found in loaded bands. "
                        f"Available: {list(band_arrays.keys())}. Skipping."
                    )
            
            if not input_bands or not target_bands:
                error_msg = (
                    f"Tri-temporal IO construction failed: "
                    f"input_bands={input_bands}, target_bands={target_bands}. "
                    f"Expected at least one 'prev_'/'mid_' band and one 'curr_' band. "
                    f"Available band_arrays keys: {list(band_arrays.keys())[:20]}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(
                f"Tri-temporal IO setup: input_bands={len(input_bands)} bands, target_bands={len(target_bands)} bands"
            )
            logger.info(f"  Input bands (T0+T1): {input_bands}")
            logger.info(f"  Target bands (T2): {target_bands}")
            
            # Build input feature vector (T0 + T1)
            x_input_list: list[np.ndarray] = []
            for b in input_bands:
                x_input_list.append(band_arrays[b][valid])
            x_input = np.stack(x_input_list, axis=1).astype(np.float32)
            
            # Build target feature vector (T2)
            y_target_list: list[np.ndarray] = []
            for b in target_bands:
                y_target_list.append(band_arrays[b][valid])
            y_target = np.stack(y_target_list, axis=1).astype(np.float32)
            
            # Sanity-check shapes
            if x_input.shape[0] != y_target.shape[0]:
                raise ValueError(
                    f"Input/target pixel count mismatch: "
                    f"x_input.shape={x_input.shape}, y_target.shape={y_target.shape}"
                )
            
            logger.info(f"Feature vectors: x_input.shape={x_input.shape}, y_target.shape={y_target.shape}")
            
            # For backward compatibility with existing code
            x_all = x_input  # Alias for compatibility
            available_bands = input_bands  # Use input bands for attention head indexing
            
            # Identify spatial and spectral band indices for Dual-Head Attention
            # These indices are relative to x_input (T0+T1 bands only)
            # Spatial: B02, B03, B04, B08 (high-resolution, structural integrity)
            # Spectral: B05, B11, B12 (physiology, spectral coherence)
            spatial_band_indices = []
            spectral_band_indices = []
            
            for idx, band_spec in enumerate(input_bands):
                band_name = band_spec.replace("prev_", "").replace("mid_", "").upper()
                
                # Spatial bands: B02, B03, B04, B08 (10m resolution, structural)
                if "B02" in band_name or "B03" in band_name or "B04" in band_name or "B08" in band_name or "NIR" in band_name:
                    spatial_band_indices.append(idx)
                
                # Spectral bands: B05, B11, B12 (physiology, moisture)
                if "B05" in band_name or "B11" in band_name or "B12" in band_name or \
                   "RED EDGE" in band_name or "RED-EDGE" in band_name or \
                   "SWIR" in band_name:
                    spectral_band_indices.append(idx)
            
            attention_head_mode = (attention_head_mode or "both").strip().lower()
            if attention_head_mode not in {"both", "spatial", "spectral"}:
                raise ValueError(
                    f"Invalid attention_head_mode: {attention_head_mode}. "
                    "Expected one of: both, spatial, spectral."
                )

            spatial_available = len(spatial_band_indices) > 0
            spectral_available = len(spectral_band_indices) > 0

            if attention_head_mode == "both" and not (spatial_available and spectral_available):
                raise ValueError(
                    "Dual-head attention requires both spatial and spectral bands. "
                    f"spatial_band_indices={spatial_band_indices}, spectral_band_indices={spectral_band_indices}. "
                    f"Input bands: {input_bands}"
                )
            if attention_head_mode == "spatial" and not spatial_available:
                raise ValueError(
                    "Spatial-only attention requested but no spatial bands found. "
                    f"spatial_band_indices={spatial_band_indices}. Input bands: {input_bands}"
                )
            if attention_head_mode == "spectral" and not spectral_available:
                raise ValueError(
                    "Spectral-only attention requested but no spectral bands found. "
                    f"spectral_band_indices={spectral_band_indices}. Input bands: {input_bands}"
                )

            if attention_head_mode == "both":
                logger.info("Attention Heads: BOTH (spatial + spectral)")
                logger.info(f"  Spatial Head (The 'Eye'): {len(spatial_band_indices)} bands at indices {spatial_band_indices}")
                logger.info(f"    Bands: {[input_bands[i] for i in spatial_band_indices]}")
                logger.info(f"  Spectral Head (The 'Lab'): {len(spectral_band_indices)} bands at indices {spectral_band_indices}")
                logger.info(f"    Bands: {[input_bands[i] for i in spectral_band_indices]}")
            elif attention_head_mode == "spatial":
                logger.info("Attention Heads: SPATIAL only")
                logger.info(f"  Spatial Head (The 'Eye'): {len(spatial_band_indices)} bands at indices {spatial_band_indices}")
                logger.info(f"    Bands: {[input_bands[i] for i in spatial_band_indices]}")
            else:
                logger.info("Attention Heads: SPECTRAL only")
                logger.info(f"  Spectral Head (The 'Lab'): {len(spectral_band_indices)} bands at indices {spectral_band_indices}")
                logger.info(f"    Bands: {[input_bands[i] for i in spectral_band_indices]}")

            # Load cluster labels if provided
            cluster_labels = None
            use_per_cluster = False
            cls = None
            cluster_summaries = None
            
            if cluster_raster_local:
                with rasterio.open(cluster_raster_local) as cluster_src:
                    # Ensure cluster raster is aligned with time_step_2 (current/reference)
                    if (
                        cluster_src.width != s2.width
                        or cluster_src.height != s2.height
                        or cluster_src.transform != s2.transform
                        or cluster_src.crs != s2.crs
                    ):
                        logger.warning(
                            "Classified raster is not aligned with time_step_2; reprojecting. "
                            f"cluster_shape={(cluster_src.height, cluster_src.width)} ts2_shape={(s2.height, s2.width)}"
                        )
                        cluster_data = cluster_src.read(1)
                        if cluster_data.ndim > 2:
                            cluster_data = cluster_data.squeeze()
                        cluster_reproj = np.full((s2.height, s2.width), 0, dtype=cluster_data.dtype)
                        reproject(
                            source=cluster_data,
                            destination=cluster_reproj,
                            src_transform=cluster_src.transform,
                            src_crs=cluster_src.crs,
                            dst_transform=s2.transform,
                            dst_crs=s2.crs,
                            resampling=Resampling.nearest,
                        )
                        cluster_labels = cluster_reproj
                    else:
                        cluster_labels = cluster_src.read(1)
                        if cluster_labels.ndim > 2:
                            cluster_labels = cluster_labels.squeeze()
                        if cluster_labels.shape != (s2.height, s2.width):
                            raise ValueError(
                                f"Cluster labels shape {cluster_labels.shape} does not match "
                                f"time_step_2 shape {(s2.height, s2.width)}"
                            )
                
                # Extract cluster labels for valid pixels only
                cluster_labels_valid = cluster_labels[valid]
                unique_clusters = np.unique(cluster_labels_valid)
                unique_clusters = unique_clusters[unique_clusters > 0]  # Exclude nodata (0)
                logger.info(f"Found {len(unique_clusters)} clusters in valid tree pixels: {sorted(unique_clusters)}")
                
                if len(unique_clusters) > 0:
                    # Process each cluster separately
                    cls_all = np.full((x_input.shape[0],), 255, dtype=np.uint8)
                    cluster_summaries = {}
                    
                    for cluster_id in unique_clusters:
                        cluster_mask = cluster_labels_valid == cluster_id
                        n_cluster_pixels = int(cluster_mask.sum())
                        if n_cluster_pixels < 10:  # Skip clusters with too few pixels
                            logger.warning(f"Skipping cluster {cluster_id} (only {n_cluster_pixels} pixels)")
                            continue
                        
                        logger.info(f"Processing cluster {cluster_id} with {n_cluster_pixels:,} pixels")
                        x_cluster_input = x_input[cluster_mask]
                        y_cluster_target = y_target[cluster_mask]
                        
                        # Train on a random subset for this cluster
                        n_train_cluster = min(max_train_pixels, x_cluster_input.shape[0])
                        idx_cluster = rng.choice(x_cluster_input.shape[0], size=n_train_cluster, replace=False)
                        x_train_cluster = x_cluster_input[idx_cluster]
                        y_train_cluster = y_cluster_target[idx_cluster]
                        
                        # Train autoencoder for this cluster with dual-head attention
                        # Verify input_dim matches feature vector
                        if x_cluster_input.shape[1] != len(input_bands):
                            raise ValueError(
                                f"Cluster {cluster_id}: Input feature vector dimension mismatch. "
                                f"x_cluster_input.shape[1]={x_cluster_input.shape[1]} but expected {len(input_bands)} "
                                f"based on input bands: {input_bands}"
                            )
                        if y_cluster_target.shape[1] != len(target_bands):
                            raise ValueError(
                                f"Cluster {cluster_id}: Target feature vector dimension mismatch. "
                                f"y_cluster_target.shape[1]={y_cluster_target.shape[1]} but expected {len(target_bands)} "
                                f"based on target bands: {target_bands}"
                            )
                        
                        # Map spatial/spectral indices to this cluster's input bands
                        model_cluster = ConvTransformerAutoencoder(
                            input_dim=x_cluster_input.shape[1],
                            target_dim=y_cluster_target.shape[1],
                            attention_head_mode=attention_head_mode,
                            spatial_band_indices=spatial_band_indices,
                            spectral_band_indices=spectral_band_indices,
                        )
                        logger.debug(f"Cluster {cluster_id}: Created autoencoder with input_dim={x_cluster_input.shape[1]}, target_dim={y_cluster_target.shape[1]}")
                        train_losses_cluster = _train(
                            model_cluster, x_train_cluster, y_train_cluster,
                            epochs=epochs, 
                            batch_size=min(batch_size, n_train_cluster), 
                            lr=lr, 
                            device=device
                        )
                        
                        train_err_cluster = _recon_error(
                            model_cluster, x_train_cluster, y_train_cluster,
                            batch_size=min(batch_size, n_train_cluster), 
                            device=device
                        )
                        
                        all_err_cluster = _recon_error(
                            model_cluster, x_cluster_input, y_cluster_target,
                            batch_size=min(batch_size, x_cluster_input.shape[0]), 
                            device=device
                        )
                        
                        # Verify reconstruction error has correct length
                        if len(all_err_cluster) != x_cluster_input.shape[0]:
                            raise ValueError(
                                f"Cluster {cluster_id}: Reconstruction error length mismatch. "
                                f"all_err_cluster.length={len(all_err_cluster)}, "
                                f"x_cluster_input.shape[0]={x_cluster_input.shape[0]}"
                            )
                        
                        # New classification strategy: Target distribution per cluster
                        # - 70% No Change (smallest errors/changes)
                        # - 5% Severe Morbidity (top 5% of total)
                        # - 10% Medium Morbidity (next 10% of total)
                        # - 15% High/Medium Vitality (remaining)
                        
                        n_cluster_total = x_cluster_input.shape[0]
                        cls_cluster = np.empty((n_cluster_total,), dtype=np.uint8)
                        
                        if no_change_percentile is not None:
                            # Step 1: Identify 70% no-change (pixels with smallest reconstruction errors)
                            no_change_threshold = float(np.quantile(all_err_cluster, no_change_percentile))
                            no_change_mask_cluster = all_err_cluster <= no_change_threshold
                            
                            # Verify mask length matches
                            if len(no_change_mask_cluster) != n_cluster_total:
                                raise ValueError(
                                    f"Cluster {cluster_id}: no_change_mask_cluster length ({len(no_change_mask_cluster)}) "
                                    f"does not match n_cluster_total ({n_cluster_total})"
                                )
                            
                            n_no_change_cluster = int(no_change_mask_cluster.sum())
                            cls_cluster[no_change_mask_cluster] = 4  # No Change
                            
                            # Step 2: For remaining 30% (changed pixels), classify by error
                            changed_mask_cluster = ~no_change_mask_cluster
                            changed_errors = all_err_cluster[changed_mask_cluster]
                            n_changed = len(changed_errors)
                            
                            # Verify changed_errors length
                            expected_changed = n_cluster_total - n_no_change_cluster
                            if n_changed != expected_changed:
                                raise ValueError(
                                    f"Cluster {cluster_id}: changed_errors length mismatch. "
                                    f"n_changed={n_changed}, expected={expected_changed} "
                                    f"(n_cluster_total={n_cluster_total}, n_no_change={n_no_change_cluster})"
                                )
                            
                            if n_changed > 0:
                                # Calculate percentiles relative to TOTAL (not just changed pixels)
                                # Top 5% of total → Severe (16.67% of changed pixels)
                                # Next 10% of total → Medium (33.33% of changed pixels)
                                # Remaining 15% of total → Vitality (50% of changed pixels)
                                
                                # Percentiles within changed pixels:
                                # - Top 16.67% of changed = top 5% of total
                                # - Next 33.33% of changed = next 10% of total
                                # - Remaining 50% of changed = remaining 15% of total
                                
                                severe_percentile_changed = 1.0 - (0.05 / (1.0 - no_change_percentile))  # Top 5% of total
                                medium_percentile_changed = severe_percentile_changed - (0.10 / (1.0 - no_change_percentile))  # Next 10% of total
                                
                                severe_q_changed = float(np.quantile(changed_errors, severe_percentile_changed))
                                medium_q_changed = float(np.quantile(changed_errors, medium_percentile_changed))
                                
                                # Classify changed pixels
                                changed_indices = np.where(changed_mask_cluster)[0]
                                
                                # Verify changed_indices length matches
                                if len(changed_indices) != n_changed:
                                    raise ValueError(
                                        f"Cluster {cluster_id}: changed_indices length mismatch. "
                                        f"len(changed_indices)={len(changed_indices)}, n_changed={n_changed}"
                                    )
                                
                                cls_changed = np.empty((n_changed,), dtype=np.uint8)
                                
                                # Severe: Top 5% of total (highest errors in changed pixels)
                                cls_changed[changed_errors > severe_q_changed] = 0
                                
                                # Medium: Next 10% of total
                                medium_mask = (changed_errors > medium_q_changed) & (changed_errors <= severe_q_changed)
                                cls_changed[medium_mask] = 1
                                
                                # Vitality: Remaining 15% of total (split into high and medium vitality)
                                vitality_mask = changed_errors <= medium_q_changed
                                vitality_errors = changed_errors[vitality_mask]
                                if len(vitality_errors) > 0:
                                    vitality_p50 = float(np.quantile(vitality_errors, 0.50))
                                    
                                    # Get indices of vitality pixels within changed_errors array
                                    vitality_indices_in_changed = np.where(vitality_mask)[0]
                                    
                                    # Split vitality pixels into high and medium vitality based on p50
                                    p50_mask_in_vitality = vitality_errors <= vitality_p50
                                    above_p50_mask_in_vitality = vitality_errors > vitality_p50
                                    
                                    # Verify masks cover all vitality pixels
                                    if p50_mask_in_vitality.sum() + above_p50_mask_in_vitality.sum() != len(vitality_errors):
                                        raise ValueError(
                                            f"Cluster {cluster_id}: Vitality p50 mask mismatch. "
                                            f"p50_mask sum={p50_mask_in_vitality.sum()}, "
                                            f"above_p50_mask sum={above_p50_mask_in_vitality.sum()}, "
                                            f"vitality_errors length={len(vitality_errors)}"
                                        )
                                    
                                    # Use indices to assign classes
                                    p50_indices = vitality_indices_in_changed[p50_mask_in_vitality]
                                    above_p50_indices = vitality_indices_in_changed[above_p50_mask_in_vitality]
                                    
                                    # Verify indices are within bounds
                                    if len(p50_indices) + len(above_p50_indices) != len(vitality_indices_in_changed):
                                        raise ValueError(
                                            f"Cluster {cluster_id}: Vitality index mismatch. "
                                            f"p50_indices={len(p50_indices)}, "
                                            f"above_p50_indices={len(above_p50_indices)}, "
                                            f"vitality_indices={len(vitality_indices_in_changed)}"
                                        )
                                    
                                    cls_changed[p50_indices] = 3  # High vitality
                                    cls_changed[above_p50_indices] = 2   # Medium vitality
                                
                                # Verify cls_changed length before assignment
                                if len(cls_changed) != len(changed_indices):
                                    raise ValueError(
                                        f"Cluster {cluster_id}: cls_changed length ({len(cls_changed)}) "
                                        f"does not match changed_indices length ({len(changed_indices)})"
                                    )
                                
                                cls_cluster[changed_indices] = cls_changed
                            
                            logger.info(f"  Cluster {cluster_id}: {n_no_change_cluster:,} pixels ({100*n_no_change_cluster/n_cluster_total:.1f}%) classified as 'no_change'")
                            logger.info(f"  Cluster {cluster_id}: Changed pixels = {n_changed:,} ({100*n_changed/n_cluster_total:.1f}%)")
                        else:
                            # Fallback to original percentile-based classification
                            thr_cluster = _thresholds(
                                all_err_cluster,
                                severe_percentile=severe_morbidity_percentile,
                                medium_percentile=medium_morbidity_percentile
                            )
                            cls_cluster[all_err_cluster <= thr_cluster.p50] = 3
                            cls_cluster[(all_err_cluster > thr_cluster.p50) & (all_err_cluster <= thr_cluster.medium_q)] = 2
                            cls_cluster[(all_err_cluster > thr_cluster.medium_q) & (all_err_cluster <= thr_cluster.severe_q)] = 1
                            cls_cluster[all_err_cluster > thr_cluster.severe_q] = 0
                        
                        # Optional: Direction-aware classification refinement (per cluster)
                        # For tri-temporal: compare T0 (oldest) vs T2 (current)
                        if use_directional_classification and ndvi0 is not None and ndvi2 is not None:
                            # Get NDVI/NDMI deltas for this cluster's pixels (T2 - T0)
                            ndvi0_valid = ndvi0[valid]
                            ndvi2_valid = ndvi2[valid]
                            
                            # Verify shapes match before computing deltas
                            if ndvi0_valid.shape != ndvi2_valid.shape:
                                logger.warning(
                                    f"Cluster {cluster_id}: NDVI shape mismatch - "
                                    f"ndvi0[valid].shape={ndvi0_valid.shape}, ndvi2[valid].shape={ndvi2_valid.shape}. "
                                    f"Skipping directional classification for this cluster."
                                )
                                ndvi_delta_cluster = None
                                ndmi_delta_cluster = None
                            else:
                                # Verify cluster_mask length matches
                                if len(cluster_mask) != len(ndvi0_valid):
                                    raise ValueError(
                                        f"Cluster {cluster_id}: cluster_mask length ({len(cluster_mask)}) "
                                        f"does not match ndvi0[valid] length ({len(ndvi0_valid)}). "
                                        f"This indicates a mismatch between cluster labels and feature vectors."
                                    )
                                ndvi_delta_cluster = ndvi2_valid[cluster_mask] - ndvi0_valid[cluster_mask]  # T2 - T0
                                ndmi_delta_cluster = None
                                if ndmi0 is not None and ndmi2 is not None:
                                    ndmi0_valid = ndmi0[valid]
                                    ndmi2_valid = ndmi2[valid]
                                    if ndmi0_valid.shape != ndmi2_valid.shape:
                                        logger.warning(
                                            f"Cluster {cluster_id}: NDMI shape mismatch - "
                                            f"ndmi0[valid].shape={ndmi0_valid.shape}, ndmi2[valid].shape={ndmi2_valid.shape}. "
                                            f"Skipping NDMI in directional classification."
                                        )
                                    elif len(cluster_mask) != len(ndmi0_valid):
                                        raise ValueError(
                                            f"Cluster {cluster_id}: cluster_mask length ({len(cluster_mask)}) "
                                            f"does not match ndmi0[valid] length ({len(ndmi0_valid)})."
                                        )
                                    else:
                                        ndmi_delta_cluster = ndmi2_valid[cluster_mask] - ndmi0_valid[cluster_mask]  # T2 - T0
                            
                            # Only apply directional classification if deltas were computed successfully
                            if ndvi_delta_cluster is not None:
                                improving_mask_cluster = ndvi_delta_cluster > directional_ndvi_threshold
                                if ndmi_delta_cluster is not None:
                                    improving_mask_cluster = improving_mask_cluster | (ndmi_delta_cluster > directional_ndmi_threshold)
                                
                                declining_mask_cluster = ndvi_delta_cluster < -directional_ndvi_threshold
                                if ndmi_delta_cluster is not None:
                                    declining_mask_cluster = declining_mask_cluster | (ndmi_delta_cluster < -directional_ndmi_threshold)
                            else:
                                # Skip directional classification if deltas couldn't be computed
                                improving_mask_cluster = np.zeros(n_cluster_pixels, dtype=bool)
                                declining_mask_cluster = np.zeros(n_cluster_pixels, dtype=bool)
                            
                            # Apply same refinement rules
                            # Verify all masks have the same length
                            if len(improving_mask_cluster) != n_cluster_pixels:
                                raise ValueError(
                                    f"Cluster {cluster_id}: improving_mask_cluster length ({len(improving_mask_cluster)}) "
                                    f"does not match n_cluster_pixels ({n_cluster_pixels})"
                                )
                            if len(declining_mask_cluster) != n_cluster_pixels:
                                raise ValueError(
                                    f"Cluster {cluster_id}: declining_mask_cluster length ({len(declining_mask_cluster)}) "
                                    f"does not match n_cluster_pixels ({n_cluster_pixels})"
                                )
                            if len(cls_cluster) != n_cluster_pixels:
                                raise ValueError(
                                    f"Cluster {cluster_id}: cls_cluster length ({len(cls_cluster)}) "
                                    f"does not match n_cluster_pixels ({n_cluster_pixels})"
                                )
                            
                            high_error_morbidity_cluster = (cls_cluster == 0) | (cls_cluster == 1)
                            improving_but_high_error_cluster = high_error_morbidity_cluster & improving_mask_cluster
                            cls_cluster[improving_but_high_error_cluster & (cls_cluster == 0)] = 1
                            cls_cluster[improving_but_high_error_cluster & (cls_cluster == 1)] = 2
                            
                            low_error_vitality_cluster = (cls_cluster == 2) | (cls_cluster == 3)
                            declining_but_low_error_cluster = low_error_vitality_cluster & declining_mask_cluster
                            cls_cluster[declining_but_low_error_cluster & (cls_cluster == 3)] = 2
                            cls_cluster[declining_but_low_error_cluster & (cls_cluster == 2)] = 1
                        
                        # Note: No-change is now handled via percentile-based classification above
                        # The old absolute delta thresholds are ignored when no_change_percentile is set
                        
                        # Store results for this cluster
                        # Verify dimensions match before assignment
                        n_cluster_mask_true = int(cluster_mask.sum())
                        if n_cluster_mask_true != len(cls_cluster):
                            raise ValueError(
                                f"Cluster {cluster_id}: Dimension mismatch when storing results. "
                                f"cluster_mask has {n_cluster_mask_true} True values, "
                                f"but cls_cluster has {len(cls_cluster)} elements. "
                                f"This indicates a mismatch in cluster processing."
                            )
                            if len(cls_all) != len(cluster_mask):
                                raise ValueError(
                                    f"Cluster {cluster_id}: cls_all length ({len(cls_all)}) "
                                    f"does not match cluster_mask length ({len(cluster_mask)}). "
                                    f"Expected both to equal number of valid pixels ({x_input.shape[0]})."
                                )
                        cls_all[cluster_mask] = cls_cluster
                        
                        # Store cluster summary
                        counts_cluster = {CLASS_NAMES[i]: int((cls_cluster == i).sum()) for i in sorted(CLASS_NAMES)}
                        rates_cluster = {k: float(v / max(1, cls_cluster.size)) for k, v in counts_cluster.items()}
                        
                        # Calculate thresholds for summary (only if not using percentile-based classification)
                        summary_thresholds = None
                        if no_change_percentile is None:
                            # Use the thresholds we calculated
                            summary_thresholds = asdict(thr_cluster)
                        else:
                            # For percentile-based, calculate thresholds from all_err_cluster for reference
                            summary_thresholds = asdict(_thresholds(
                                all_err_cluster,
                                severe_percentile=severe_morbidity_percentile,
                                medium_percentile=medium_morbidity_percentile
                            ))
                        
                        cluster_summaries[int(cluster_id)] = {
                            "n_pixels": n_cluster_pixels,
                            "train_pixels": int(n_train_cluster),
                            "thresholds": summary_thresholds,
                            "error_train_mean": float(np.mean(train_err_cluster)),
                            "error_all_mean": float(np.mean(all_err_cluster)),
                            "class_counts": counts_cluster,
                            "class_rates": rates_cluster,
                        }
                    
                    # After processing all clusters, decide whether to use per-cluster results
                    if len(cluster_summaries) > 0:
                        cls = cls_all
                        use_per_cluster = True
                        logger.info(f"Completed per-cluster processing for {len(cluster_summaries)} clusters")
                    else:
                        logger.warning("No valid clusters found after filtering, falling back to global autoencoder")
            else:
                logger.warning("No valid clusters found, falling back to global autoencoder")
            
            # Global autoencoder approach (used when no cluster raster provided OR when falling back from per-cluster)
            if not use_per_cluster:
                # Global autoencoder approach: train on a random subset
                n_train = min(max_train_pixels, x_input.shape[0])
                idx = rng.choice(x_input.shape[0], size=n_train, replace=False)
                x_train = x_input[idx]
                y_train = y_target[idx]

                # Verify dimensions
                if x_input.shape[1] != len(input_bands):
                    raise ValueError(
                        f"Input feature vector dimension mismatch: x_input.shape[1]={x_input.shape[1]} "
                        f"but expected {len(input_bands)} based on input bands: {input_bands}"
                    )
                if y_target.shape[1] != len(target_bands):
                    raise ValueError(
                        f"Target feature vector dimension mismatch: y_target.shape[1]={y_target.shape[1]} "
                        f"but expected {len(target_bands)} based on target bands: {target_bands}"
                    )
                
                model = ConvTransformerAutoencoder(
                    input_dim=x_input.shape[1],
                    target_dim=y_target.shape[1],
                    attention_head_mode=attention_head_mode,
                    spatial_band_indices=spatial_band_indices,
                    spectral_band_indices=spectral_band_indices,
                )
                logger.info(f"Created autoencoder with input_dim={x_input.shape[1]}, target_dim={y_target.shape[1]}")
                logger.info(f"  Input bands (T0+T1): {input_bands}")
                logger.info(f"  Target bands (T2): {target_bands}")
                train_losses = _train(model, x_train, y_train, epochs=epochs, batch_size=min(batch_size, n_train), lr=lr, device=device)

                all_err = _recon_error(model, x_input, y_target, batch_size=min(batch_size, x_input.shape[0]), device=device)

                # New classification strategy: Target distribution
                # - 70% No Change (smallest errors)
                # - 5% Severe Morbidity (top 5% of total)
                # - 10% Medium Morbidity (next 10% of total)
                # - 15% High/Medium Vitality (remaining)
                
                n_total = x_input.shape[0]
                cls = np.empty((n_total,), dtype=np.uint8)
                
                if no_change_percentile is not None:
                    # Step 1: Identify 70% no-change (pixels with smallest reconstruction errors)
                    no_change_threshold = float(np.quantile(all_err, no_change_percentile))
                    no_change_mask = all_err <= no_change_threshold
                    n_no_change = int(no_change_mask.sum())
                    cls[no_change_mask] = 4  # No Change
                    
                    # Step 2: For remaining 30% (changed pixels), classify by error
                    changed_mask = ~no_change_mask
                    changed_errors = all_err[changed_mask]
                    n_changed = len(changed_errors)
                    
                    if n_changed > 0:
                        # Calculate percentiles relative to TOTAL
                        # Top 5% of total → Severe (16.67% of changed pixels)
                        # Next 10% of total → Medium (33.33% of changed pixels)
                        # Remaining 15% of total → Vitality (50% of changed pixels)
                        
                        severe_percentile_changed = 1.0 - (0.05 / (1.0 - no_change_percentile))
                        medium_percentile_changed = severe_percentile_changed - (0.10 / (1.0 - no_change_percentile))
                        
                        severe_q_changed = float(np.quantile(changed_errors, severe_percentile_changed))
                        medium_q_changed = float(np.quantile(changed_errors, medium_percentile_changed))
                        
                        # Classify changed pixels
                        changed_indices = np.where(changed_mask)[0]
                        cls_changed = np.empty((n_changed,), dtype=np.uint8)
                        
                        # Severe: Top 5% of total
                        cls_changed[changed_errors > severe_q_changed] = 0
                        
                        # Medium: Next 10% of total
                        medium_mask = (changed_errors > medium_q_changed) & (changed_errors <= severe_q_changed)
                        cls_changed[medium_mask] = 1
                        
                        # Vitality: Remaining 15% of total
                        vitality_mask = changed_errors <= medium_q_changed
                        vitality_errors = changed_errors[vitality_mask]
                        if len(vitality_errors) > 0:
                            vitality_p50 = float(np.quantile(vitality_errors, 0.50))
                            cls_changed[vitality_mask & (vitality_errors <= vitality_p50)] = 3  # High vitality
                            cls_changed[vitality_mask & (vitality_errors > vitality_p50)] = 2   # Medium vitality
                        
                        cls[changed_indices] = cls_changed
                    
                    logger.info(f"No-change pixels: {n_no_change:,} ({100*n_no_change/n_total:.1f}% of total)")
                    logger.info(f"Changed pixels: {n_changed:,} ({100*n_changed/n_total:.1f}% of total)")
                else:
                    # Fallback to original percentile-based classification
                    thr = _thresholds(
                        all_err,
                        severe_percentile=severe_morbidity_percentile,
                        medium_percentile=medium_morbidity_percentile
                    )
                    cls[all_err <= thr.p50] = 3
                    cls[(all_err > thr.p50) & (all_err <= thr.medium_q)] = 2
                    cls[(all_err > thr.medium_q) & (all_err <= thr.severe_q)] = 1
                    cls[all_err > thr.severe_q] = 0

                # Optional: Direction-aware classification refinement
                # This addresses the limitation that autoencoder doesn't check direction of change
                # For tri-temporal: compare T0 (oldest) vs T2 (current)
                if use_directional_classification and ndvi0 is not None and ndvi2 is not None:
                    # Calculate directional deltas for valid pixels (T2 - T0, positive = improvement)
                    ndvi_delta = ndvi2[valid] - ndvi0[valid]
                    ndmi_delta = None
                    if ndmi0 is not None and ndmi2 is not None:
                        ndmi_delta = ndmi2[valid] - ndmi0[valid]
                    
                    # Refine classification based on direction:
                    # - High error + declining NDVI/NDMI → confirm morbidity (keep as is)
                    # - High error + improving NDVI/NDMI → might be recovery, downgrade severity
                    # - Low error + improving NDVI/NDMI → confirm vitality (keep as is)
                    # - Low error + declining NDVI/NDMI → might be false positive, upgrade to morbidity
                    
                    # Identify improving vs declining pixels
                    improving_mask = ndvi_delta > directional_ndvi_threshold
                    if ndmi_delta is not None:
                        improving_mask = improving_mask | (ndmi_delta > directional_ndmi_threshold)
                    
                    declining_mask = ndvi_delta < -directional_ndvi_threshold
                    if ndmi_delta is not None:
                        declining_mask = declining_mask | (ndmi_delta < -directional_ndmi_threshold)
                    
                    # Refinement rules:
                    # 1. High error (morbidity) but improving → downgrade to medium morbidity or medium vitality
                    high_error_morbidity = (cls == 0) | (cls == 1)  # severe or medium morbidity
                    improving_but_high_error = high_error_morbidity & improving_mask
                    cls[improving_but_high_error & (cls == 0)] = 1  # severe → medium morbidity
                    cls[improving_but_high_error & (cls == 1)] = 2  # medium morbidity → medium vitality
                    
                    # 2. Low error (vitality) but declining → upgrade to medium morbidity
                    low_error_vitality = (cls == 2) | (cls == 3)  # medium or high vitality
                    declining_but_low_error = low_error_vitality & declining_mask
                    cls[declining_but_low_error & (cls == 3)] = 2  # high vitality → medium vitality
                    cls[declining_but_low_error & (cls == 2)] = 1  # medium vitality → medium morbidity
                    
                    logger.info(f"Direction-aware refinement: {improving_but_high_error.sum()} improving pixels downgraded, {declining_but_low_error.sum()} declining pixels upgraded")

                # Note: No-change is now handled via percentile-based classification above
                # The old absolute delta thresholds are ignored when no_change_percentile is set
                
                # Initialize cluster_summaries if not already set
                if 'cluster_summaries' not in locals():
                    cluster_summaries = None

            # Rehydrate to raster (use T2 as reference)
            class_raster = np.full((s2.height, s2.width), 255, dtype=np.uint8)
            class_raster[valid] = cls

            # Save outputs locally
            out_tif = os.path.join(tmp, "vitality_classes.tif")
            prof = s2.profile.copy()  # Use T2 (current) as reference
            prof.update(count=1, dtype=rasterio.uint8, nodata=255, compress="deflate", predictor=2)
            with rasterio.open(out_tif, "w", **prof) as dst:
                dst.write(class_raster, 1)
                dst.set_band_description(1, "VITALITY_CLASS")
                # Encode class labels + a simple colormap to make the raster easier to interpret in GIS tools.
                dst.update_tags(
                    1,
                    **{f"CLASS_{k}": v for k, v in sorted(CLASS_NAMES.items())},
                )
                dst.update_tags(
                    CLASS_NAMES=json.dumps(CLASS_NAMES, sort_keys=True),
                )
                dst.write_colormap(
                    1,
                    {
                        0: (215, 25, 28, 255),      # severe_morbidity (red)
                        1: (253, 174, 97, 255),     # medium_morbidity (orange)
                        2: (255, 255, 191, 255),    # medium_vitality (yellow)
                        3: (26, 150, 65, 255),      # high_vitality (green)
                        4: (128, 128, 128, 255),    # no_change (gray)
                    },
                )

            # Save metrics JSON
            counts = {CLASS_NAMES[i]: int((cls == i).sum()) for i in sorted(CLASS_NAMES)}
            rates = {k: float(v / max(1, cls.size)) for k, v in counts.items()}
            
            # Log morbidity percentages
            severe_pct = rates.get("severe_morbidity", 0.0) * 100
            medium_pct = rates.get("medium_morbidity", 0.0) * 100
            no_change_pct = rates.get("no_change", 0.0) * 100
            
            logger.info(f"💚 Vitality Analysis Results for tile {tile_id}:")
            logger.info(f"   Severe Morbidity Threshold: {severe_morbidity_percentile*100:.1f}th percentile (target: 5%)")
            logger.info(f"   Medium Morbidity Threshold: {medium_morbidity_percentile*100:.1f}th percentile (target: 10% between {medium_morbidity_percentile*100:.0f}th-{severe_morbidity_percentile*100:.0f}th)")
            logger.info(f"   High Morbidity: {severe_pct:.2f}%")
            logger.info(f"   Medium Morbidity: {medium_pct:.2f}%")
            logger.info(f"   Total Morbidity (High + Medium): {severe_pct + medium_pct:.2f}%")
            logger.info(f"   No Change: {no_change_pct:.2f}%")
            
            # Validate percentages and issue warnings (but continue processing)
            warnings_list = []
            
            if no_change_percentile is not None:
                # New percentile-based classification: Target 70% no-change, 5% severe, 10% medium, 15% vitality
                expected_no_change = no_change_percentile * 100
                expected_severe = 5.0
                expected_medium = 10.0
                expected_vitality = 15.0
                
                no_change_tolerance = 3.0
                if abs(no_change_pct - expected_no_change) > no_change_tolerance:
                    warnings_list.append(
                        f"⚠️  No Change: {no_change_pct:.2f}% (expected ~{expected_no_change:.1f}% "
                        f"for {no_change_percentile*100:.0f}th percentile). "
                        f"Classification may be incorrect."
                    )
                
                severe_tolerance = 1.0
                if abs(severe_pct - expected_severe) > severe_tolerance:
                    warnings_list.append(
                        f"⚠️  Severe Morbidity: {severe_pct:.2f}% (expected ~{expected_severe:.1f}%). "
                        f"Classification may be incorrect."
                    )
                
                medium_tolerance = 2.0
                if abs(medium_pct - expected_medium) > medium_tolerance:
                    warnings_list.append(
                        f"⚠️  Medium Morbidity: {medium_pct:.2f}% (expected ~{expected_medium:.1f}%). "
                        f"Classification may be incorrect."
                    )
            else:
                # Old threshold-based classification: Check medium morbidity
                expected_medium_morbidity = (severe_morbidity_percentile - medium_morbidity_percentile) * 100
                medium_morbidity_tolerance = 3.0
                if abs(medium_pct - expected_medium_morbidity) > medium_morbidity_tolerance:
                    warnings_list.append(
                        f"⚠️  Medium Morbidity: {medium_pct:.2f}% (expected ~{expected_medium_morbidity:.1f}% "
                        f"for {medium_morbidity_percentile*100:.0f}th-{severe_morbidity_percentile*100:.0f}th percentile range). "
                        f"Threshold calculation may be incorrect."
                    )
                
                # Check no change: should be > 0% (with default thresholds)
                if no_change_pct == 0.0 and (no_change_abs_delta_ndvi is not None or no_change_abs_delta_ndmi is not None):
                    warnings_list.append(
                        f"⚠️  No Change: {no_change_pct:.2f}% (expected > 0%). "
                        f"No pixels classified as 'no_change' despite thresholds being set "
                        f"(NDVI: {no_change_abs_delta_ndvi}, NDMI: {no_change_abs_delta_ndmi}). "
                        f"Thresholds may be too strict or NDVI/NDMI data may be missing."
                    )
            
            # Log warnings if any
            if warnings_list:
                logger.warning(f"⚠️  Vitality Classification Warnings for tile {tile_id}:")
                for warning in warnings_list:
                    logger.warning(f"   {warning}")
                logger.warning(f"   ⚠️  Continuing with other tiles...")
            else:
                logger.info(f"   ✅ All percentages within expected ranges")
            
            if cluster_summaries is not None:
                # Per-cluster processing summary
                summary = {
                    "tile_id": tile_id,
                    "n_valid_pixels": n_valid,
                    "epochs": int(epochs),
                    "lr": float(lr),
                    "device": str(device),
                    "no_change_abs_delta_ndvi": float(no_change_abs_delta_ndvi) if no_change_abs_delta_ndvi is not None else None,
                    "no_change_abs_delta_ndmi": float(no_change_abs_delta_ndmi) if no_change_abs_delta_ndmi is not None else None,
                    "processing_mode": "per_cluster",
                    "n_clusters": len(cluster_summaries),
                    "clusters": cluster_summaries,
                    "class_counts": counts,
                    "class_rates": rates,
                }
            else:
                # Global processing summary
                summary = {
                    "tile_id": tile_id,
                    "n_valid_pixels": n_valid,
                    "train_pixels": int(n_train),
                    "epochs": int(epochs),
                    "lr": float(lr),
                    "device": str(device),
                    "no_change_abs_delta_ndvi": float(no_change_abs_delta_ndvi) if no_change_abs_delta_ndvi is not None else None,
                    "no_change_abs_delta_ndmi": float(no_change_abs_delta_ndmi) if no_change_abs_delta_ndmi is not None else None,
                    "no_change_percentile": float(no_change_percentile) if no_change_percentile is not None else None,
                    "processing_mode": "global",
                    "train_mse_first": float(train_losses[0]) if train_losses else None,
                    "train_mse_last": float(train_losses[-1]) if train_losses else None,
                    "thresholds": asdict(thr),
                    "error_train_mean": float(np.mean(train_err)),
                    "error_train_p95": float(np.quantile(train_err, 0.95)),
                    "error_all_mean": float(np.mean(all_err)),
                    "error_all_p95": float(np.quantile(all_err, 0.95)),
                    "class_counts": counts,
                    "class_rates": rates,
                }
            out_json = os.path.join(tmp, "summary.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

        os.makedirs(output_s3_dir, exist_ok=True)
        final_tif = _join(output_s3_dir, "vitality_classes.tif")
        final_json = _join(output_s3_dir, "summary.json")
        os.replace(out_tif, final_tif)
        os.replace(out_json, final_json)
        logger.info(f"Saved:\n - {final_tif}\n - {final_json}")
        return {"vitality_classes_tif": final_tif, "summary_json": final_json}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pixel-based vitality via autoencoder over (prev,curr) NDMI/NDVI")
    p.add_argument("--base-output-dir", required=True, help="Local base output directory for a processed AOI run")
    p.add_argument("--tile-id", required=True, help="e.g. 32UQV")
    p.add_argument("--output-s3-dir", default=None)
    p.add_argument("--epochs", type=int, default=24, help="Number of training epochs (default: 24 for Conv-Transformer)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=262144)
    p.add_argument("--max-train-pixels", type=int, default=300000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-change-abs-delta-ndvi",
        type=float,
        default=0.03,
        help=(
            "Threshold for 'no_change' class: pixels with abs(curr_ndvi - prev_ndvi) <= this are classified as 'no_change'. "
            "NDVI is scaled to [-1, 1]. Default: 0.03 (3% change threshold)."
        ),
    )
    p.add_argument(
        "--no-change-abs-delta-ndmi",
        type=float,
        default=0.03,
        help=(
            "Threshold for 'no_change' class: pixels with abs(curr_ndmi - prev_ndmi) <= this are classified as 'no_change'. "
            "NDMI is scaled to [-1, 1]. Default: 0.03 (3% change threshold). "
            "Note: Ignored if --no-change-percentile is set."
        ),
    )
    p.add_argument(
        "--no-change-percentile",
        type=float,
        default=0.70,
        help=(
            "Percentile-based no-change classification: Bottom X% of pixels by reconstruction error are classified as 'no_change'. "
            "Default: 0.70 (70% no-change). When set, this overrides --no-change-abs-delta-ndvi/ndmi thresholds. "
            "Remaining pixels are classified as: 5% severe, 10% medium morbidity, 15% vitality."
        ),
    )
    p.add_argument(
        "--classified-raster-s3",
        type=str,
        default=None,
        help=(
            "Optional: S3 path to k-means classified raster (e.g., from segmentation step). "
            "If provided, trains a separate autoencoder per cluster for better outlier sensitivity."
        ),
    )
    p.add_argument(
        "--severe-morbidity-percentile",
        type=float,
        default=0.95,
        help=(
            "Percentile threshold for severe morbidity classification (default: 0.95, i.e., top 5%). "
            "Higher values (e.g., 0.98) result in fewer severe morbidity pixels. "
            "Lower values (e.g., 0.90) result in more severe morbidity pixels."
        ),
    )
    p.add_argument(
        "--medium-morbidity-percentile",
        type=float,
        default=0.85,
        help=(
            "Percentile threshold for medium morbidity classification (default: 0.85, i.e., 85th percentile). "
            "Together with severe_morbidity_percentile (0.95), this creates a 10% range for medium morbidity "
            "(between 85th-95th percentile). Higher values result in fewer medium morbidity pixels."
        ),
    )
    p.add_argument(
        "--feature-bands",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Bands to use for autoencoder features. Format: 'prev_NDVI', 'curr_NDMI', etc. "
            "Default: ['prev_NDMI', 'prev_NDVI', 'curr_NDMI', 'curr_NDVI']. "
            "Available bands: NDVI, NDMI, B02, B03, B04, B05, B06, B08, B11, B12. "
            "Prefix with 'prev_' for previous time step, 'curr_' for current time step."
        ),
    )
    p.add_argument(
        "--use-directional-classification",
        action="store_true",
        default=True,
        help=(
            "Enable direction-aware classification refinement. "
            "Combines reconstruction error with NDVI/NDMI direction of change. "
            "High error + declining NDVI/NDMI → confirm morbidity. "
            "High error + improving NDVI/NDMI → downgrade severity. "
            "Low error + improving NDVI/NDMI → confirm vitality. "
            "Low error + declining NDVI/NDMI → upgrade to morbidity."
        ),
    )
    p.add_argument(
        "--directional-ndvi-threshold",
        type=float,
        default=0.05,
        help=(
            "NDVI delta threshold for directional classification (default: 0.05). "
            "Pixels with |curr_NDVI - prev_NDVI| > this are considered 'changing'. "
            "Used only if --use-directional-classification is enabled."
        ),
    )
    p.add_argument(
        "--directional-ndmi-threshold",
        type=float,
        default=0.05,
        help=(
            "NDMI delta threshold for directional classification (default: 0.05). "
            "Pixels with |curr_NDMI - prev_NDMI| > this are considered 'changing'. "
            "Used only if --use-directional-classification is enabled."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_tile(
        base_output_dir=args.base_output_dir,
        tile_id=args.tile_id,
        output_s3_dir=args.output_s3_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_pixels=args.max_train_pixels,
        seed=args.seed,
        no_change_abs_delta_ndvi=args.no_change_abs_delta_ndvi,
        no_change_abs_delta_ndmi=args.no_change_abs_delta_ndmi,
        no_change_percentile=getattr(args, "no_change_percentile", 0.70),
        classified_raster_s3=args.classified_raster_s3,
        severe_morbidity_percentile=args.severe_morbidity_percentile,
        medium_morbidity_percentile=args.medium_morbidity_percentile,
        feature_bands=args.feature_bands,
        use_directional_classification=args.use_directional_classification,
        directional_ndvi_threshold=args.directional_ndvi_threshold,
        directional_ndmi_threshold=args.directional_ndmi_threshold,
    )


if __name__ == "__main__":
    main()


