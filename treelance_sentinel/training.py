"""
Training script for UNet-based classification model with enhanced error handling and GPU support.

FIXES APPLIED:
1. Fixed test data splitting logic - now handles cases where all data is labeled
2. Enhanced GPU detection with detailed CUDA status logging
3. Added data quality checks (size, class balance)
4. Improved error handling for BatchNorm initialization
5. Better logging throughout the training process
6. Fallback to validation data when no separate test data exists

USAGE:
    poetry run python training.py

REQUIREMENTS:
    - PyTorch with CUDA support for GPU training
    - NVIDIA drivers properly installed
    - Sufficient labeled data (recommended: >1000 samples)
"""

# Ensure PROJ and GDAL use the active conda environment's data directories
import os
_conda_prefix = os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV")
if _conda_prefix:
    _proj_path = os.path.join(_conda_prefix, "share", "proj")
    _gdal_path = os.path.join(_conda_prefix, "share", "gdal")
    os.environ.setdefault("PROJ_DATA", _proj_path)
    os.environ.setdefault("PROJ_LIB", _proj_path)
    os.environ.setdefault("GDAL_DATA", _gdal_path)

import geopandas as gpd 
import torch
import torch.nn as nn
import torch.optim as optim
import math
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from numpy import mean, std
from sklearn.preprocessing import StandardScaler
import pandas as pd
from loguru import logger
import time
import os
from treelance_sentinel.utils import timer_decorator, Timer, setup_logger
from tqdm import tqdm
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import shutil

setup_logger()

# File paths configuration
PATHS = {
    # Default input: user-labeled zonal stats with new band set
    'input_gpkg': "data/training/S2A_32UQV_20240709_0_L2A_stack_B02-B03-B04-B05-B06-B08-B11-B12_clipped_zonal_stats.gpkg",
    # Base model output path (latest checkpoint). F1-tagged copies will be derived from this.
    'model_output': "model_checkpoints/sentinel_unet_s2a_32uqv_20240709.pt",
    'raster_output': 'local_output/training/classification.tif',
    'binary_raster_output': 'local_output/training/binary_classification.tif',
}


def _next_retrain_suffix_path(path: str) -> str:
    """
    Given a model checkpoint path (local or S3), return a new path with an
    incremented `_retrain_N` suffix before the extension.
    
    Examples:
        base.pt            -> base_retrain_1.pt
        base_retrain_1.pt  -> base_retrain_2.pt
    """
    if not path:
        return path

    # Helper to build new filename with incremented suffix
    def _bump_name(filename: str) -> str:
        root, ext = os.path.splitext(filename)
        m = re.search(r"(.*)_retrain_(\d+)$", root)
        if m:
            prefix = m.group(1)
            idx = int(m.group(2)) + 1
        else:
            prefix = root
            idx = 1
        return f"{prefix}_retrain_{idx}{ext}"

    dir_path, filename = os.path.split(path)
    new_filename = _bump_name(filename)
    return os.path.join(dir_path, new_filename)

class AttentionModule(nn.Module):
    """Multi-head self-attention mechanism for tabular data."""
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(AttentionModule, self).__init__()
        # Ensure num_heads divides input_dim evenly, or use a smaller number
        if input_dim < num_heads:
            num_heads = 1
        elif input_dim % num_heads != 0:
            # Find the largest divisor less than or equal to num_heads
            for i in range(num_heads, 0, -1):
                if input_dim % i == 0:
                    num_heads = i
                    break
        
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # For tabular data, treat features as sequence dimension
        # Reshape from [batch_size, features] to [batch_size, 1, features]
        x_reshaped = x.unsqueeze(1)  # Add sequence dimension
        
        # Multi-head attention
        Q = self.query(x_reshaped).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x_reshaped).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x_reshaped).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.num_heads * self.head_dim)
        
        # Remove sequence dimension and project
        context = context.squeeze(1)  # Back to [batch_size, features]
        output = self.proj(context)
        output = self.layer_norm(output + x)
        return output

class AcrossFeatureAttention(nn.Module):
    """True across-feature self-attention for tabular vectors.

    Treats each feature as a token and applies MHSA over the feature axis.
    For input x of shape [batch, num_features], we create per-feature token
    embeddings of size d_model, run attention across the feature dimension,
    then project back to a vector of size num_features with a per-feature
    projection, followed by a residual connection and LayerNorm.
    """
    def __init__(self, num_features, d_model=64, num_heads=4, dropout=0.1):
        super(AcrossFeatureAttention, self).__init__()
        if d_model % num_heads != 0:
            # Reduce heads to a divisor of d_model for safety
            for h in range(num_heads, 0, -1):
                if d_model % h == 0:
                    num_heads = h
                    break
        self.num_features = num_features
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Per-feature token embedding parameters
        # tokens = x[..., None] * feature_emb + feature_bias
        self.feature_emb = nn.Parameter(torch.randn(num_features, d_model) * (1.0 / math.sqrt(d_model)))
        self.feature_bias = nn.Parameter(torch.zeros(num_features, d_model))

        # Attention projections applied to the token channel (last dim)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj_tokens = nn.Linear(d_model, d_model)

        # Project tokens back to a scalar per feature with per-feature weights
        self.out_proj_features = nn.Parameter(torch.randn(num_features, d_model) * (1.0 / math.sqrt(d_model)))
        self.out_bias_features = nn.Parameter(torch.zeros(num_features))

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_features)
        # For interpretability stats
        self.collect_stats = False
        self._attn_sum = None
        self._attn_batches = 0

    def forward(self, x):
        # x: [B, C]
        B, C = x.shape
        assert C == self.num_features, f"AcrossFeatureAttention expected {self.num_features} features, got {C}"

        # Create tokens of shape [B, C, D]
        tokens = x.unsqueeze(-1) * self.feature_emb + self.feature_bias

        # Compute Q, K, V: [B, C, 3D]
        qkv = self.qkv_proj(tokens)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape to heads: [B, C, H, Dh] -> [B, H, C, Dh]
        def split_heads(t):
            return t.view(B, C, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Attention over feature axis: [B, H, C, Dh] x [B, H, Dh, C] -> [B, H, C, C]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        if self.collect_stats:
            # Average across batch and heads -> [C, C], accumulate
            avg_attn = attn.mean(dim=(0, 1))
            if self._attn_sum is None or self._attn_sum.shape != avg_attn.shape or self._attn_sum.device != avg_attn.device:
                self._attn_sum = torch.zeros_like(avg_attn)
                self._attn_batches = 0
            self._attn_sum = self._attn_sum + avg_attn.detach()
            self._attn_batches += 1

        # Context: [B, H, C, Dh]
        context = torch.matmul(attn, v)
        # Merge heads: [B, C, H, Dh] -> [B, C, D]
        context = context.transpose(1, 2).contiguous().view(B, C, self.d_model)
        context = self.out_proj_tokens(context)
        context = self.proj_dropout(context)

        # Project tokens back to scalar per feature with per-feature weights
        # out[b, c] = sum_d context[b, c, d] * W_feat[c, d] + b_feat[c]
        out = (context * self.out_proj_features).sum(dim=-1) + self.out_bias_features

        # Residual over feature vector with LayerNorm across features
        out = self.layer_norm(out + x)
        return out

    def reset_stats(self):
        self._attn_sum = None
        self._attn_batches = 0

    def get_average_attention(self):
        if self._attn_sum is None or self._attn_batches == 0:
            return None
        return (self._attn_sum / float(self._attn_batches)).detach().cpu().numpy()

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

class DimensionChangeResidualBlock(nn.Module):
    """Residual block that can handle dimension changes with a projection layer."""
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.1):
        super(DimensionChangeResidualBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim)
        
        # Add in_features and out_features attributes for PyTorch compatibility
        self.in_features = input_dim
        self.out_features = output_dim
        
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        # Projection layer for residual connection if dimensions differ
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.block(x)
        
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
            
        out += residual
        return self.relu(out)

# Band order when first 8 inputs are band means (b02_mean .. b12_mean):
# [B02, B03, B04, B05, B06, B08, B11, B12] -> indices 0..7.
# Primary classification goal: differentiate grass/agriculture vs trees (urban is easier to separate).
# Red-edge (B05, B06) and SWIR (B11, B12) best separate woody canopy from herbaceous vegetation.
TREE_GRASS_PRIORITY_BAND_INDICES = [3, 4, 6, 7]  # B05, B06, B11, B12 — tree vs grass/agriculture
TREE_GRASS_INIT_PRIORITY_SCALE = 1.25  # Slightly stronger prior so attention focuses on these bands


class InputBandBias(nn.Module):
    """
    Input scaling so attention focuses on bands that separate trees from grass/agriculture.
    Urban is typically easy to separate; the hard distinction is woody (tree) vs herbaceous (grass/crops).
    Use when the first N inputs are band means in order [B02, B03, B04, B05, B06, B08, B11, B12].
    priority_indices: 0-based indices to upweight — [3,4,6,7] = B05, B06, B11, B12 (red-edge + SWIR).
    """
    def __init__(self, input_size: int, priority_indices: list[int] | None = None, init_priority_scale: float | None = None):
        super().__init__()
        if init_priority_scale is None:
            init_priority_scale = TREE_GRASS_INIT_PRIORITY_SCALE
        self.scale = nn.Parameter(torch.ones(input_size))
        if priority_indices is not None:
            with torch.no_grad():
                for i in priority_indices:
                    if 0 <= i < input_size:
                        self.scale[i] = init_priority_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.view(1, -1)


class SpectralAttention1D(nn.Module):
    """
    Spectral Attention over projected band features (1D zonal stats).
    
    Primary goal: differentiate grass/agriculture vs trees. Urban is easier to separate.
    Learnable channel weights let the model emphasize:
    - Red-edge (B05, B06): chlorophyll slope and structure — woody vs herbaceous.
    - SWIR (B11, B12): moisture and structure — forest vs grassland/crops.
    NIR (B08) and VIS (B02–B04) still help for urban vs vegetation; the input band bias
    already upweights B05, B06, B11, B12 so this head can refine tree vs grass separation.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Channel-wise attention for 1D feature vectors (operates on projected space)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Learnable per-channel weights (initialized neutral; model learns tree vs grass emphasis)
        self.band_bias = nn.Parameter(torch.ones(channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels) - 1D feature vectors
        Returns:
            Attention-weighted tensor of same shape
        """
        # Channel attention: (batch, channels) -> (batch, channels)
        y = self.fc(x)
        
        # Apply learned spectral weights + band priorities
        return x * y * self.band_bias.view(1, -1)


class DeepUNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        base_channels=48,
        use_spectral_attention: bool = True,
        tree_grass_band_priority_indices: list[int] | None = None,
    ):
        super(DeepUNetEncoder, self).__init__()
        # Focus attention on tree vs grass/agriculture: upweight B05, B06, B11, B12 (red-edge + SWIR)
        self.input_band_bias = (
            InputBandBias(input_channels, priority_indices=tree_grass_band_priority_indices)
            if tree_grass_band_priority_indices
            else None
        )
        # Initial feature extraction with attention (increased capacity)
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, base_channels * 2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        # Across-feature attention directly on input features for interpretability
        self.input_attention = AcrossFeatureAttention(num_features=input_channels, d_model=128, num_heads=8)  # Increased d_model and heads
        
        # Spectral Attention: learnable channel weighting after projection (tree vs grass focus).
        if use_spectral_attention:
            self.spectral_attention = SpectralAttention1D(base_channels * 2)
        else:
            self.spectral_attention = None
        
        # Multiple encoder blocks with increasing complexity
        self.enc1 = nn.Sequential(
            DimensionChangeResidualBlock(base_channels * 2, base_channels * 4),
            AcrossFeatureAttention(num_features=base_channels * 4, d_model=128, num_heads=8),  # Increased capacity
            nn.Linear(base_channels * 4, base_channels * 4),  # Keep same dimension
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.enc2 = nn.Sequential(
            DimensionChangeResidualBlock(base_channels * 4, base_channels * 8),
            AcrossFeatureAttention(num_features=base_channels * 8, d_model=128, num_heads=8),  # Increased capacity
            nn.Linear(base_channels * 8, base_channels * 8),  # Keep same dimension
            nn.BatchNorm1d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Add enc3 back for more capacity with additional features
        self.enc3 = nn.Sequential(
            DimensionChangeResidualBlock(base_channels * 8, base_channels * 12),
            AcrossFeatureAttention(num_features=base_channels * 12, d_model=128, num_heads=8),
            nn.Linear(base_channels * 12, base_channels * 12),
            nn.BatchNorm1d(base_channels * 12),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        if self.input_band_bias is not None:
            x = self.input_band_bias(x)
        # Collect attention over raw features for interpretability (no change to downstream dims)
        x_in_attended = self.input_attention(x)
        x0 = self.input_proj(x_in_attended)
        
        # Apply spectral attention if enabled (tree vs grass/agriculture)
        if self.spectral_attention is not None:
            x0 = self.spectral_attention(x0)
        
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)  # Added back for more capacity
        return x0, x1, x2, x3

class DeepUNetDecoder(nn.Module):
    def __init__(self, base_channels=48):  # Increased from 10 to 48 to match encoder
        super(DeepUNetDecoder, self).__init__()
        
        # Decoder blocks with skip connections - Increased to match encoder depth
        self.dec3 = nn.Sequential(
            DimensionChangeResidualBlock(base_channels * 12, base_channels * 8),   # 576 → 384
            AcrossFeatureAttention(num_features=base_channels * 8, d_model=128, num_heads=8),  # Increased capacity
            nn.Linear(base_channels * 8, base_channels * 8),  # Keep 384
            nn.BatchNorm1d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.dec2 = nn.Sequential(
            DimensionChangeResidualBlock(base_channels * 8 + base_channels * 8, base_channels * 4),   # 384+384=768 → 192
            AcrossFeatureAttention(num_features=base_channels * 4, d_model=128, num_heads=8),  # Increased capacity
            nn.Linear(base_channels * 4, base_channels * 4),  # Keep 192
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.dec1 = nn.Sequential(
            DimensionChangeResidualBlock(base_channels * 4 + base_channels * 4, base_channels * 2),    # 192+192=384 → 96
            AcrossFeatureAttention(num_features=base_channels * 2, d_model=128, num_heads=8),  # Increased capacity
            nn.Linear(base_channels * 2, base_channels * 2),  # Keep 96
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, x0, x1, x2, x3):
        # Decode with skip connections (increased depth to match encoder)
        # Start from x3 (576 features)
        
        # Debug: Log tensor shapes during decoding
        if hasattr(self, 'debug_shapes') and self.debug_shapes:
            logger.info(f"Decoder - x3: {x3.shape}")
        
        # Process x3: 576 → 384 features
        d3 = self.dec3(x3)  # 576 → 384
        
        if hasattr(self, 'debug_shapes') and self.debug_shapes:
            logger.info(f"Decoder - d3: {d3.shape}, x2: {x2.shape}")
        
        # Concatenate d3 (384) with x2 (384) = 768 features
        d2 = self.dec2(torch.cat([d3, x2], dim=1))  # Skip connection: 768 → 192
        
        if hasattr(self, 'debug_shapes') and self.debug_shapes:
            logger.info(f"Decoder - d2: {d2.shape}, x1: {x1.shape}")
        
        # Concatenate d2 (192) with x1 (192) = 384 features
        d1 = self.dec1(torch.cat([d2, x1], dim=1))  # Skip connection: 384 → 96
        return d1

class DeepUNetClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        base_channels=48,
        use_spectral_attention: bool = True,
        tree_grass_band_priority_indices: list[int] | None = None,
    ):
        super(DeepUNetClassifier, self).__init__()
        self.input_size = input_size
        self.base_channels = base_channels
        # When first 8 inputs are band means [B02..B12], pass tree_grass_band_priority_indices to
        # focus attention on tree vs grass/agriculture (B05, B06, B11, B12).
        self.encoder = DeepUNetEncoder(
            input_size,
            base_channels,
            use_spectral_attention=use_spectral_attention,
            tree_grass_band_priority_indices=tree_grass_band_priority_indices,
        )
        self.decoder = DeepUNetDecoder(base_channels)
        
        # Final classification layers (increased capacity for more features)
        self.final = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels * 3),  # Increased from base_channels * 2 to * 3
            nn.BatchNorm1d(base_channels * 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_channels * 3, base_channels * 2),  # Additional layer for more capacity
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(base_channels * 2, num_classes)
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encode with multiple levels (increased depth for more features)
        x0, x1, x2, x3 = self.encoder(x)
        
        # Debug: Log tensor shapes
        if hasattr(self, 'debug_shapes') and self.debug_shapes:
            logger.info(f"Encoder outputs - x0: {x0.shape}, x1: {x1.shape}, x2: {x2.shape}, x3: {x3.shape}")
        
        # Decode with skip connections (increased depth to match encoder)
        decoded = self.decoder(x0, x1, x2, x3)
     
        
        if hasattr(self, 'debug_shapes') and self.debug_shapes:
            logger.info(f"Decoder output: {decoded.shape}")
        
        # Final classification
        return self.final(decoded)

# Keep the old UNetClassifier for backward compatibility
class UNetClassifier(DeepUNetClassifier):
    def __init__(self, input_size, num_classes, tree_grass_band_priority_indices: list[int] | None = None):
        super().__init__(
            input_size,
            num_classes,
            base_channels=33,
            tree_grass_band_priority_indices=tree_grass_band_priority_indices,
        )  # Set to 33 to target ~1M parameters

class GeoDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, device, scheduler=None, num_epochs=2, patience=15, checkpoint_extra=None, model_output_path=None, max_grad_norm=1.0, s3_model_path=None):
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    logger.info(f"Training: {len(train_loader.dataset)} samples, {num_epochs} epochs, patience {patience}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for features, labels in train_loader_tqdm:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            batch_count += 1
            train_loader_tqdm.set_postfix(loss=loss.item())
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_labels = []
        all_val_preds = []
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for features, labels in val_loader_tqdm:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        val_precision = precision_score(all_val_labels, all_val_preds, average='macro')
        val_recall = recall_score(all_val_labels, all_val_preds, average='macro')
        
        # Step LR scheduler on validation loss
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Log epoch results (minimal)
        logger.info(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} Acc {train_acc:.1f}% | Val Loss {avg_val_loss:.4f} Acc {val_acc:.1f}% F1 {val_f1:.4f}")
        
        # Save best model based on F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            # Get model's input size and number of classes
            # Use the actual feature count, not the encoder layer dimensions
            input_size = getattr(model, 'input_size', None)
            if input_size is None:
                try:
                    input_size = model.encoder.input_proj[0].in_features
                except Exception:
                    input_size = None
            # Resolve number of classes from the final classifier layer
            try:
                num_classes = model.final[-1].out_features
            except Exception:
                # Fallback if final is not a Sequential or indexing fails
                num_classes = None
            
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'input_size': input_size,
                'num_classes': num_classes
            }
            # ALWAYS include scaler statistics for production use
            if checkpoint_extra is not None:
                try:
                    ckpt.update(checkpoint_extra)
                except Exception:
                    pass
            # Ensure scaler is always saved (critical for inference on new data)
            if 'scaler_mean' not in ckpt or ckpt['scaler_mean'] is None:
                if checkpoint_extra and 'scaler_mean' in checkpoint_extra:
                    ckpt['scaler_mean'] = checkpoint_extra['scaler_mean']
                    ckpt['scaler_scale'] = checkpoint_extra['scaler_scale']
                    logger.info("Scaler statistics added to checkpoint")
            model_path = model_output_path if model_output_path else PATHS['model_output']
            torch.save(ckpt, model_path)
            logger.info(f"New best model saved locally! F1: {val_f1:.4f} (Epoch {epoch+1})")
        else:
            patience_counter += 1
            if patience_counter % 5 == 0:  # Only log every 5 epochs of no improvement
                logger.info(f"No improvement for {patience_counter} epochs. Best F1: {best_val_f1:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                logger.info(f"Best model was from epoch {best_epoch + 1} with validation F1: {best_val_f1:.4f}")
                break
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr}")
        logger.info("-" * 50)
    
    # Load and return the best model
    logger.info(f"Training completed. Loading best model from epoch {best_epoch + 1}")
    try:
        model_path = model_output_path if model_output_path else PATHS['model_output']
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Best model loaded successfully")
        return model
    except Exception as e:
        logger.warning(f"Failed to load best model: {e}. Returning current model state.")
        return model

@timer_decorator
def process_training_data():
    try:
        logger.info("Starting training data processing")
        logger.info("Rasterizing geometries by classification")
        
        start_time = time.perf_counter()
        # Rasterization logic here
        logger.info(f"\033[92m⏱️ Rasterization completed in {time.perf_counter() - start_time:.2f} seconds\033[0m")
        
    except Exception as e:
        logger.error(f"Failed to process training data: {str(e)}")
        logger.exception("Full traceback:")
        raise

def plot_confusion_matrices(y_true, y_pred, class_names, output_dir, dataset_name="Validation"):
    """Generate and save confusion matrices with per-class metrics summary."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Overall confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    # Convert to percentages
    cm_percentage = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_percentage, interpolation='nearest', cmap='YlOrRd', vmin=0, vmax=100)
    plt.title(f'{dataset_name} Set - Overall Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    plt.colorbar(label='Percentage')
    
    # Add text annotations with percentages and counts
    thresh = 50  # Threshold for text color
    for i in range(cm_percentage.shape[0]):
        for j in range(cm_percentage.shape[1]):
            plt.text(j, i, f'{cm_percentage[i, j]:.1f}%\n({int(cm[i, j])})',
                    ha="center", va="center",
                    color="white" if cm_percentage[i, j] > thresh else "black",
                    fontsize=10, fontweight='bold')
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha='right')
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_overall_{dataset_name.lower()}_{timestamp}.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate per-class metrics
    per_class_metrics = []
    
    # Per-class confusion matrices
    for class_idx, class_name in enumerate(class_names):
        # Create binary confusion matrix for this class
        binary_cm = np.zeros((2, 2))
        binary_cm[0, 0] = sum((y_true != class_idx) & (y_pred != class_idx))  # True Negatives
        binary_cm[0, 1] = sum((y_true != class_idx) & (y_pred == class_idx))  # False Positives
        binary_cm[1, 0] = sum((y_true == class_idx) & (y_pred != class_idx))  # False Negatives
        binary_cm[1, 1] = sum((y_true == class_idx) & (y_pred == class_idx))  # True Positives
        
        # Convert to percentages
        binary_cm_percentage = binary_cm.astype('float') / (binary_cm.sum(axis=1)[:, np.newaxis] + 1e-8) * 100
        
        plt.figure(figsize=(8, 6))
        plt.imshow(binary_cm_percentage, interpolation='nearest', cmap='YlOrRd', vmin=0, vmax=100)
        plt.title(f'{dataset_name} Set - Confusion Matrix for {class_name} (Percentages)', fontsize=12, fontweight='bold')
        plt.colorbar(label='Percentage')
        
        # Add text annotations with percentages
        thresh = 50  # Threshold for text color
        for i in range(binary_cm_percentage.shape[0]):
            for j in range(binary_cm_percentage.shape[1]):
                plt.text(j, i, f'{binary_cm_percentage[i, j]:.1f}%\n({int(binary_cm[i, j])})',
                        ha="center", va="center",
                        color="white" if binary_cm_percentage[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')
        
        plt.ylabel('True Label', fontsize=11)
        plt.xlabel('Predicted Label', fontsize=11)
        plt.xticks([0, 1], ['Not ' + class_name, class_name])
        plt.yticks([0, 1], ['Not ' + class_name, class_name])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{class_name}_{dataset_name.lower()}_{timestamp}.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics for this class
        tn, fp, fn, tp = binary_cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(tp + fn)  # Number of true instances of this class
        
        per_class_metrics.append({
            'Class': class_name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support,
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn)
        })
    
    # Create and save per-class metrics summary table
    metrics_df = pd.DataFrame(per_class_metrics)
    
    # Log detailed metrics
    logger.info(f"\n{'='*80}")
    logger.info(f"{dataset_name} Set - Per-Class Metrics Summary")
    logger.info(f"{'='*80}")
    logger.info(f"\n{metrics_df.to_string(index=False)}")
    logger.info(f"\n{'='*80}")
    
    # Save metrics to CSV
    csv_path = os.path.join(output_dir, f'per_class_metrics_{dataset_name.lower()}_{timestamp}.csv')
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Per-class metrics saved to: {csv_path}")
    
    return metrics_df

# ============================================================================
# SPATIAL SPLITTING FUNCTIONS (Critical Fix for Spatial Leakage)
# ============================================================================

def spatial_block_split(X, y, data_train, test_size=0.1, val_size=0.2, strategy='utm_zone'):
    """
    Split data spatially to prevent spatial leakage.
    
    Strategies:
    - 'utm_zone': Split by UTM zone_number (groups all samples from same zone together)
    - 'spatial_grid': Create 10km x 10km grid blocks and split by block
    - 'polygon_id': If polygon/cluster IDs exist, use GroupKFold
    
    Args:
        X: Feature matrix (DataFrame)
        y: Target vector (Series)
        data_train: Full training DataFrame (needed for spatial columns)
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining after test)
        strategy: Splitting strategy ('utm_zone', 'spatial_grid', 'polygon_id')
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info(f"Using spatial splitting strategy: {strategy}")
    
    if strategy == 'utm_zone' and 'utm_zone_number' in data_train.columns:
        # Split by UTM zone - all samples from same zone stay together
        logger.info("Splitting by UTM zone_number to prevent spatial leakage...")
        unique_zones = data_train['utm_zone_number'].unique()
        n_zones = len(unique_zones)
        
        # If only one zone, fall back to polygon ID or spatial grid
        if n_zones == 1:
            logger.warning(f"All data is in a single UTM zone ({unique_zones[0]}).")
            logger.warning("Cannot split by zone. Falling back to polygon ID or spatial grid...")
            
            # Try polygon ID first
            if 'ID' in data_train.columns:
                logger.info("Falling back to polygon ID splitting...")
                strategy = 'polygon_id'
            else:
                logger.warning("No polygon ID available. Using spatial grid fallback...")
                # Create a simple spatial grid based on geometry centroids
                logger.info("Creating spatial grid for splitting...")
                centroids = data_train.geometry.centroid
                # Create grid cells (roughly 10km x 10km)
                grid_size = 10000  # 10km in meters
                grid_x = (centroids.x // grid_size).astype(int)
                grid_y = (centroids.y // grid_size).astype(int)
                grid_id = grid_x * 10000 + grid_y  # Create unique grid cell ID
                
                unique_grids = grid_id.unique()
                n_grids = len(unique_grids)
                n_test_grids = max(1, int(n_grids * test_size))
                n_val_grids = max(1, int((n_grids - n_test_grids) * val_size))
                
                np.random.seed(42)
                shuffled_grids = np.random.permutation(unique_grids)
                test_grids = set(shuffled_grids[:n_test_grids])
                val_grids = set(shuffled_grids[n_test_grids:n_test_grids + n_val_grids])
                train_grids = set(shuffled_grids[n_test_grids + n_val_grids:])
                
                test_mask = grid_id.isin(test_grids)
                val_mask = grid_id.isin(val_grids)
                train_mask = grid_id.isin(train_grids)
                
                X_test = X[test_mask].copy()
                y_test = y[test_mask].copy()
                X_val = X[val_mask].copy()
                y_val = y[val_mask].copy()
                X_train = X[train_mask].copy()
                y_train = y[train_mask].copy()
                
                logger.info(f"Spatial grid split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                logger.info(f"Grid cells - Train: {len(train_grids)}, Val: {len(val_grids)}, Test: {len(test_grids)}")
                
                return X_train, X_val, X_test, y_train, y_val, y_test
        
        else:
            # Multiple zones - split by zone
            n_test_zones = max(1, int(n_zones * test_size))
            n_val_zones = max(1, int((n_zones - n_test_zones) * val_size))
            
            # Shuffle zones (not samples) to randomize which zones go to test/val
            np.random.seed(42)
            shuffled_zones = np.random.permutation(unique_zones)
            test_zones = set(shuffled_zones[:n_test_zones])
            val_zones = set(shuffled_zones[n_test_zones:n_test_zones + n_val_zones])
            train_zones = set(shuffled_zones[n_test_zones + n_val_zones:])
            
            logger.info(f"Zones split: {len(train_zones)} train, {len(val_zones)} val, {len(test_zones)} test")
            
            # Create masks based on zone membership
            test_mask = data_train['utm_zone_number'].isin(test_zones)
            val_mask = data_train['utm_zone_number'].isin(val_zones)
            train_mask = data_train['utm_zone_number'].isin(train_zones)
            
            X_test = X[test_mask].copy()
            y_test = y[test_mask].copy()
            X_val = X[val_mask].copy()
            y_val = y[val_mask].copy()
            X_train = X[train_mask].copy()
            y_train = y[train_mask].copy()
            
            logger.info(f"Spatial split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            logger.info(f"Zone distribution - Train: {sorted(train_zones)}, Val: {sorted(val_zones)}, Test: {sorted(test_zones)}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
    
    if strategy == 'polygon_id' and 'ID' in data_train.columns:
        # Use polygon/cluster IDs for grouping
        logger.info("Splitting by polygon/cluster ID to prevent spatial leakage...")
        unique_ids = data_train['ID'].unique()
        n_ids = len(unique_ids)
        n_test_ids = max(1, int(n_ids * test_size))
        n_val_ids = max(1, int((n_ids - n_test_ids) * val_size))
        
        np.random.seed(42)
        shuffled_ids = np.random.permutation(unique_ids)
        test_ids = set(shuffled_ids[:n_test_ids])
        val_ids = set(shuffled_ids[n_test_ids:n_test_ids + n_val_ids])
        train_ids = set(shuffled_ids[n_test_ids + n_val_ids:])
        
        test_mask = data_train['ID'].isin(test_ids)
        val_mask = data_train['ID'].isin(val_ids)
        train_mask = data_train['ID'].isin(train_ids)
        
        X_test = X[test_mask].copy()
        y_test = y[test_mask].copy()
        X_val = X[val_mask].copy()
        y_val = y[val_mask].copy()
        X_train = X[train_mask].copy()
        y_train = y[train_mask].copy()
        
        logger.info(f"Spatial split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    else:
        # Fallback: if no spatial columns available, warn and use random split
        logger.warning(f"Spatial splitting strategy '{strategy}' not available. Required columns missing.")
        logger.warning("Falling back to RANDOM split (WARNING: May cause spatial leakage!)")
        logger.warning("Consider adding utm_zone_number or ID column for proper spatial splitting.")
        
        X_pool, X_test, y_pool, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_pool, y_pool, test_size=val_size, random_state=42, stratify=y_pool
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================================
# XGBOOST BENCHMARKING
# ============================================================================

def train_xgboost_baseline(X_train, y_train, X_val, y_val, X_test, y_test, feature_columns):
    """
    Train XGBoost as a baseline to compare against Deep U-Net.
    
    Returns:
        dict with model, predictions, and metrics
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost not installed. Skipping baseline benchmark.")
        return None
    
    logger.info("=" * 60)
    logger.info("Training XGBoost Baseline (for comparison)")
    logger.info("=" * 60)
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'  # Faster training
    )
    
    start_time = time.time()
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - start_time
    
    # Predictions
    y_train_pred = xgb_model.predict(X_train)
    y_val_pred = xgb_model.predict(X_val)
    y_test_pred = xgb_model.predict(X_test)
    
    # Probabilities for calibration
    y_train_proba = xgb_model.predict_proba(X_train)
    y_val_proba = xgb_model.predict_proba(X_val)
    y_test_proba = xgb_model.predict_proba(X_test)
    
    # Metrics
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    logger.info(f"XGBoost Results:")
    logger.info(f"  Train F1: {train_f1:.4f}, Acc: {train_acc:.4f}")
    logger.info(f"  Val F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
    logger.info(f"  Test F1: {test_f1:.4f}, Acc: {test_acc:.4f}")
    logger.info(f"  Training time: {train_time:.2f}s")
    
    return {
        'model': xgb_model,
        'train_pred': y_train_pred,
        'val_pred': y_val_pred,
        'test_pred': y_test_pred,
        'train_proba': y_train_proba,
        'val_proba': y_val_proba,
        'test_proba': y_test_proba,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'test_f1': test_f1,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_time': train_time
    }

# ============================================================================
# PROBABILITY CALIBRATION EVALUATION
# ============================================================================

def evaluate_calibration(y_true, y_proba, n_bins=10):
    """
    Evaluate probability calibration using Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (n_samples, n_classes)
        n_bins: Number of bins for calibration curve
    
    Returns:
        dict with ECE, calibration curve data, and calibration plot
    """
    from sklearn.calibration import calibration_curve
    
    n_classes = y_proba.shape[1]
    ece_scores = []
    calibration_data = {}
    
    for class_idx in range(n_classes):
        # Binary calibration for each class (one-vs-rest)
        y_true_binary = (y_true == class_idx).astype(int)
        y_proba_binary = y_proba[:, class_idx]
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_proba_binary, n_bins=n_bins, strategy='uniform'
        )
        
        # Calculate ECE (Expected Calibration Error)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper, frac_pos, mean_pred in zip(
            bin_lowers, bin_uppers, fraction_of_positives, mean_predicted_value
        ):
            # Find samples in this bin
            in_bin = (y_proba_binary > bin_lower) & (y_proba_binary <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                ece += np.abs(frac_pos - mean_pred) * prop_in_bin
        
        ece_scores.append(ece)
        calibration_data[class_idx] = {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value,
            'ece': ece
        }
    
    mean_ece = np.mean(ece_scores)
    
    logger.info(f"Calibration Evaluation (ECE):")
    logger.info(f"  Mean ECE: {mean_ece:.4f}")
    logger.info(f"  Per-class ECE: {ece_scores}")
    logger.info(f"  Interpretation: ECE < 0.05 = well calibrated, ECE > 0.15 = poorly calibrated")
    
    return {
        'mean_ece': mean_ece,
        'per_class_ece': ece_scores,
        'calibration_data': calibration_data
    }

def main():
    """Main training function that will only run when this file is executed directly."""
    import argparse
    import yaml
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='UNet Classification Model Training/Retraining')
    parser.add_argument('--retrain', action='store_true', help='Retrain existing model instead of training from scratch')
    parser.add_argument('--model-path', type=str, help='Path to existing model checkpoint for retraining')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args, unknown = parser.parse_known_args()
    
    # Load config file if provided and override PATHS
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        training_config = config.get('training', {})
        if training_config.get('input_gpkg'):
            PATHS['input_gpkg'] = training_config['input_gpkg']
            logger.info(f"Using input_gpkg from config: {PATHS['input_gpkg']}")
        if training_config.get('model_output_path'):
            PATHS['model_output'] = training_config['model_output_path']
            logger.info(f"Using model_output_path from config: {PATHS['model_output']}")
    elif args.config:
        logger.warning(f"Config file specified but not found: {args.config}. Using default PATHS.")
    
    temp_dir = None
    temp_paths = {}  # Track which paths are temporary
    local_paths = {}  # Map original paths to local paths
    
    try:
        local_paths['input_gpkg'] = PATHS['input_gpkg']
        if args.model_path:
            local_paths['model_path'] = args.model_path

        local_paths['model_output'] = PATHS['model_output']
        os.makedirs(os.path.dirname(local_paths['model_output']), exist_ok=True)

        local_paths['raster_output'] = PATHS['raster_output']
        os.makedirs(os.path.dirname(local_paths['raster_output']), exist_ok=True)

        local_paths['binary_raster_output'] = PATHS['binary_raster_output']
        os.makedirs(os.path.dirname(local_paths['binary_raster_output']), exist_ok=True)
        
        # Replace the original file path references
        input_path = local_paths['input_gpkg']
        logger.info(f"Reading training data from {PATHS['input_gpkg']}")

        # Convert GPKG to Parquet if input is GPKG (more efficient for training)
        if input_path.endswith('.gpkg'):
            logger.info("Input is GPKG format. Converting to Parquet for efficient training...")
            try:
                import pandas as pd
                
                # Read GPKG
                logger.info(f"Reading GPKG file: {input_path}")
                gdf_temp = gpd.read_file(input_path)
                
                parquet_local = input_path.replace('.gpkg', '.parquet')
                
                # Convert to pandas DataFrame (drop geometry for Parquet)
                logger.info(f"Converting GPKG to Parquet: {parquet_local}")
                df_temp = gdf_temp.drop(columns=['geometry']) if 'geometry' in gdf_temp.columns else gdf_temp
                df_temp.to_parquet(parquet_local, index=False)
                logger.info(f"Successfully converted to Parquet: {parquet_local}")
                
                # Use Parquet file for training
                input_path = parquet_local
                logger.info(f"Using converted Parquet file for training: {parquet_local}")
            except Exception as conv_err:
                logger.warning(f"Failed to convert GPKG to Parquet: {conv_err}. Reading GPKG directly.")
                logger.warning(f"Error details: {conv_err}", exc_info=True)
                # Fallback to reading GPKG directly
                pass

        try:
            # Read the file - support both GPKG and Parquet formats
            if input_path.endswith('.parquet'):
                import pandas as pd
                df = pd.read_parquet(input_path)
                # If it's Parquet, it might not have geometry (dropped during conversion)
                # For rasterization, we need geometry, so try to get it from original GPKG if available
                if 'geometry' in df.columns:
                    logger.info("Parquet file contains geometry column, keeping it for rasterization")
                    gdf = gpd.GeoDataFrame(df)
                else:
                    logger.info("Parquet file does not contain geometry column")
                    # Try to read geometry from original GPKG file if it exists
                    gpkg_path = input_path.replace('.parquet', '.gpkg')
                    if os.path.exists(gpkg_path):
                        logger.info(f"Reading geometry from original GPKG: {gpkg_path}")
                        gdf_geom = gpd.read_file(gpkg_path, columns=['geometry'])
                        # Merge geometry back if indices match
                        if len(df) == len(gdf_geom):
                            gdf = gpd.GeoDataFrame(df, geometry=gdf_geom.geometry.values)
                            logger.info("Successfully merged geometry from GPKG")
                        else:
                            logger.warning(f"Geometry length mismatch: {len(df)} vs {len(gdf_geom)}, creating GeoDataFrame without geometry")
                            gdf = gpd.GeoDataFrame(df)
                    else:
                        logger.warning("No geometry available - rasterization will be skipped")
                        # Create a GeoDataFrame without geometry for compatibility
                        gdf = gpd.GeoDataFrame(df)
            else:
                # Read as GPKG (fallback case)
                logger.info("Reading GPKG file directly (conversion to Parquet may have failed)")
                gdf = gpd.read_file(input_path)
                # Keep geometry for rasterization later, but it won't be used in training features
                if 'geometry' in gdf.columns:
                    logger.info("Keeping geometry column for later rasterization (will be excluded from features)")
            logger.info(f"Training data file successfully read ({len(gdf)} rows).")
        except Exception as e:
            logger.error(f"Failed to read training data file: {e}")
            raise

        # Split the GeoDataFrame into labeled and unlabeled
        logger.info("Splitting GeoDataFrame into labeled and unlabeled by 'class_id'.")
        labeled_mask = ~gdf.class_id.isnull()
        unlabeled_mask = gdf.class_id.isnull()
        
        logger.info(f"Total rows: {len(gdf)}")
        logger.info(f"Labeled rows: {labeled_mask.sum()}")
        logger.info(f"Unlabeled rows: {unlabeled_mask.sum()}")
        
        # Check if we have any labeled data
        if labeled_mask.sum() == 0:
            logger.error("=" * 60)
            logger.error("CRITICAL: No labeled data found in the input file!")
            logger.error("=" * 60)
            logger.error("The training script requires labeled data (non-null 'class_id' values).")
            logger.error("Please ensure your input GPKG file contains labeled polygons.")
            logger.error("")
            logger.error("To create labeled data:")
            logger.error("1. Run the prediction workflow to generate confidence outputs")
            logger.error("2. Manually label some polygons based on the confidence maps")
            logger.error("3. Update the GPKG file with class_id values (0=grassland, 1=tree, 2=urban)")
            logger.error("=" * 60)
            raise ValueError("No labeled data found. Cannot proceed with training.")
        
        data_train = gdf.loc[labeled_mask].copy()

        # Fill missing values with 0 in training data
        logger.info("Filling missing values with 0 in training dataset.")
        data_train.fillna(0, inplace=True)
        
        # Note: Unlabeled data (if any) will be used later only for inference, not for evaluation

        # Dynamically derive feature columns from available zonal stats
        # Pattern: band-code-based 'bXX_mean' / 'bXX_std' (preferred) or legacy index-based 'b{idx}_mean'.
        count_present = 'count' in data_train.columns
        band_features = {}
        for col in data_train.columns:
            if col.startswith('b') and ('_mean' in col or '_std' in col):
                try:
                    band_str, stat = col.split('_', 1)
                    # Support both 'b1_mean' and 'b02_mean' styles. Extract numeric part after 'b'.
                    num_str = band_str[1:]
                    idx = int(num_str)
                except Exception:
                    continue
                band_features.setdefault(idx, set()).add(stat)

        # Build ordered list: ONLY band means (no std, no count)
        feature_columns = []
        for idx in sorted(band_features.keys()):
            # Prefer band-code style 'bXX_mean' (e.g. b02_mean), but accept legacy 'b{idx}_mean' if that's all we have.
            band_code_col = f"b{idx:02d}_mean"
            legacy_col = f"b{idx}_mean"
            if band_code_col in data_train.columns:
                feature_columns.append(band_code_col)
            elif legacy_col in data_train.columns:
                feature_columns.append(legacy_col)

        if not feature_columns:
            raise ValueError("No feature columns discovered matching pattern 'b*_mean'.")

        # Add UTM zone features if present (split into three categorical features)
        # EPSG:32632 = 3 (first digit) + 26 (middle two digits, North) + 32 (last two digits, zone)
        if 'utm_first_digit' in data_train.columns:
            feature_columns.append('utm_first_digit')
            logger.info("Including UTM first digit as a feature (always 3 for UTM)")
        if 'utm_hemisphere_code' in data_train.columns:
            feature_columns.append('utm_hemisphere_code')
            logger.info("Including UTM hemisphere code as a feature (26=North, 27=South)")
        if 'utm_zone_number' in data_train.columns:
            feature_columns.append('utm_zone_number')
            logger.info("Including UTM zone number as a feature (preserves spatial proximity: adjacent zones have close values)")
        # Backward compatibility: also check for old columns
        if 'utm_hemisphere' in data_train.columns and 'utm_hemisphere_code' not in data_train.columns:
            feature_columns.append('utm_hemisphere')
            logger.info("Including legacy UTM hemisphere as a feature (1=North, 0=South)")
        if 'utm_zone' in data_train.columns and 'utm_zone_number' not in data_train.columns:
            feature_columns.append('utm_zone')
            logger.info("Including legacy UTM zone (full EPSG code) as a feature")
        if 'utm_zone_number' not in data_train.columns and 'utm_zone' not in data_train.columns:
            logger.warning("UTM zone columns not found in training data. They will not be included as features.")
        
        # H3 geo embeddings removed: hash-based H3 loses spatial proximity and creates
        # thousands of categorical values with no benefit. UTM zone_number already provides
        # spatial proximity (adjacent zones have close numeric values).
        # If H3 columns are present, they will be ignored (not added to feature_columns).
        
        # Add year and month as temporal features (if present)
        if 'year' in data_train.columns:
            feature_columns.append('year')
            logger.info("Including year as a temporal feature")
        else:
            logger.warning("Year column not found in training data. It will not be included as a feature.")
        
        if 'month' in data_train.columns:
            feature_columns.append('month')
            logger.info("Including month as a temporal feature")
        else:
            logger.warning("Month column not found in training data. It will not be included as a feature.")

        logger.info(f"Using feature columns (n={len(feature_columns)}): {feature_columns[:12]}{' ...' if len(feature_columns)>12 else ''}")

        X = data_train[feature_columns]
        y = data_train['class_id'].astype(np.int64)  # Ensure class_id is int64

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector distribution:\n{y.value_counts()}")
        
        # Check if we have enough data for training
        if len(X) < 100:
            logger.warning(f"Very small training set: {len(X)} samples. This may lead to poor model performance.")
        elif len(X) < 1000:
            logger.info(f"Training set size: {len(X)} samples. Consider collecting more data for better performance.")
        else:
            logger.info(f"Training set size: {len(X)} samples. Good amount of data for training.")
        
        # Check class balance
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
        
        if imbalance_ratio > 10:
            logger.warning(f"Severe class imbalance detected: ratio = {imbalance_ratio:.1f}")
            logger.info("Consider using class weights or data augmentation techniques.")
        elif imbalance_ratio > 3:
            logger.info(f"Moderate class imbalance: ratio = {imbalance_ratio:.1f}")
        else:
            logger.info("Classes are reasonably balanced.")

        # CRITICAL FIX: Use spatial splitting to prevent spatial leakage
        # Instead of random shuffle, split by spatial location (UTM zone or polygon ID)
        logger.info("=" * 60)
        logger.info("SPATIAL SPLITTING (Preventing Spatial Leakage)")
        logger.info("=" * 60)
        logger.info("Using spatial block splitting instead of random shuffle.")
        logger.info("This prevents training samples from being adjacent to test samples,")
        logger.info("which would inflate test metrics by 10-15%.")
        
        # Try spatial splitting strategies in order of preference
        split_strategy = 'utm_zone'  # Default: split by UTM zone
        if 'utm_zone_number' not in data_train.columns and 'ID' in data_train.columns:
            split_strategy = 'polygon_id'
            logger.info("UTM zone_number not available, using polygon ID for spatial splitting")
        elif 'utm_zone_number' not in data_train.columns:
            logger.warning("No spatial columns available (utm_zone_number or ID).")
            logger.warning("Falling back to random split - SPATIAL LEAKAGE RISK!")
            split_strategy = 'random'
        
        if split_strategy != 'random':
            X_train, X_val, X_test_base, y_train, y_val, y_test_base = spatial_block_split(
                X, y, data_train, test_size=0.1, val_size=0.2, strategy=split_strategy
            )
        else:
            # Fallback to random (with warning)
            logger.warning("Using RANDOM split - this may cause spatial leakage!")
        X_pool, X_test_base, y_pool, y_test_base = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_pool, y_pool, test_size=0.2, random_state=42, stratify=y_pool
        )
        
        logger.info(f"Training set shape (pre-undersampling): {X_train.shape}")
        logger.info(f"Validation set shape: {X_val.shape}")
        logger.info(f"Test set shape (10%): {X_test_base.shape}")

        # Log class distribution before undersampling
        logger.info("\nClass distribution before undersampling:")
        logger.info(pd.Series(y_train).value_counts().sort_index())

        # Identify categorical features - these should NOT be standardized
        # Categorical features include:
        # - UTM first_digit: Always 3 (constant, but kept for completeness)
        # - UTM hemisphere_code: 26 (North) or 27 (South) - discrete choice, not spatially ordered
        # 
        # Note: H3 geo embeddings removed - hash-based H3 loses spatial proximity and creates
        # thousands of categorical values with no benefit. UTM zone_number already provides
        # spatial proximity (adjacent zones have close numeric values).
        # 
        # Numerical features (NOT standardized to preserve spatial proximity):
        # - UTM zone_number: Adjacent zones (32, 33, 34) have close numeric values
        #                    Treating as numerical preserves spatial proximity
        categorical_features = []
        if 'utm_first_digit' in feature_columns:
            categorical_features.append('utm_first_digit')
        if 'utm_hemisphere_code' in feature_columns:
            categorical_features.append('utm_hemisphere_code')
        # Backward compatibility: old columns
        if 'utm_hemisphere' in feature_columns:
            categorical_features.append('utm_hemisphere')
        if 'utm_zone' in feature_columns:
            categorical_features.append('utm_zone')
        
        # Numerical features that preserve spatial proximity (NOT standardized)
        # utm_zone_number is kept as raw numerical to preserve spatial proximity
        # (adjacent zones 32, 33, 34 have close numeric values)
        numerical_spatial_features = []
        if 'utm_zone_number' in feature_columns:
            numerical_spatial_features.append('utm_zone_number')
        
        # Separate continuous (standardized) and categorical features
        # Note: numerical_spatial_features are NOT in categorical_features, so they won't be standardized
        # but they're also not in continuous_features, so they'll remain as raw numerical values
        continuous_features = [col for col in feature_columns 
                               if col not in categorical_features and col not in numerical_spatial_features]
        
        if numerical_spatial_features:
            logger.info(f"Numerical spatial features (not standardized, preserve spatial proximity): {numerical_spatial_features}")
        
        logger.info(f"Categorical features (not standardized): {categorical_features}")
        logger.info(f"Continuous features (standardized): {len(continuous_features)} features")

        # Train-only standardization (reuse scaler from checkpoint when retraining)
        # Only standardize continuous features; categorical features remain as-is
        logger.info("Setting up StandardScaler for continuous features only (reuse from checkpoint if retraining)...")
        scaler = StandardScaler()
        scaler_loaded = False
        if args.retrain:
            model_path_for_scaler = local_paths.get('model_path') or local_paths.get('model_output') or (args.model_path if args.model_path else local_paths['model_output'])
            if os.path.exists(model_path_for_scaler):
                try:
                    ckpt_scaler = torch.load(model_path_for_scaler, map_location=torch.device('cpu'))
                    mean_ = ckpt_scaler.get('scaler_mean', None)
                    scale_ = ckpt_scaler.get('scaler_scale', None)
                    if mean_ is not None and scale_ is not None:
                        scaler.mean_ = np.array(mean_)
                        scaler.scale_ = np.array(scale_)
                        scaler.var_ = scaler.scale_ ** 2
                        scaler_loaded = True
                        logger.info("Using StandardScaler statistics from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load scaler from checkpoint: {e}")

        if not scaler_loaded:
            logger.info("Fitting StandardScaler on continuous features in train split only...")
            scaler.fit(X_train[continuous_features])

        # Standardize only continuous features; keep categorical and numerical spatial features as-is
        # - Categorical features: remain as-is (not standardized)
        # - Numerical spatial features (utm_zone_number): remain as raw values (preserves spatial proximity)
        # - Continuous features: standardized (mean=0, std=1)
        X_test_base_scaled = X_test_base.copy()
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        
        if continuous_features:
            X_test_base_scaled[continuous_features] = scaler.transform(X_test_base[continuous_features])
            X_train_scaled[continuous_features] = scaler.transform(X_train[continuous_features])
            X_val_scaled[continuous_features] = scaler.transform(X_val[continuous_features])
        
        # Convert back to DataFrames to preserve column names
        X_test_base_scaled = pd.DataFrame(X_test_base_scaled, index=X_test_base.index, columns=X_test_base.columns)
        X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)

        # Apply SMOTE for oversampling to balance classes (oversample minority classes to match majority)
        logger.info("Applying SMOTE to handle class imbalance on training split...")
        logger.info("SMOTE will oversample minority classes to match the majority class size.")
        try:
            train_indices = X_train_scaled.index.to_numpy()
            X_train_array = X_train_scaled.values
            y_train_array = y_train.reset_index(drop=True).values
            
            # Apply SMOTE to oversample minority classes to match majority class
            smote = SMOTE(
                random_state=42,
                k_neighbors=min(5, min(pd.Series(y_train_array).value_counts()) - 1),  # Adaptive k_neighbors
                sampling_strategy='auto'  # Oversample all minority classes to match majority
            )
            try:
                X_res_array, y_res_array = smote.fit_resample(X_train_array, y_train_array)
                logger.info(f"After SMOTE: {X_res_array.shape[0]} samples (was {X_train_array.shape[0]})")
                logger.info(f"Class distribution after SMOTE:\n{pd.Series(y_res_array).value_counts().sort_index()}")
            except Exception as smote_err:
                logger.warning(f"SMOTE failed (likely insufficient samples): {smote_err}")
                logger.info("Falling back to original data without SMOTE")
                X_res_array, y_res_array = X_train_array, y_train_array
            
            # Convert back to DataFrame with proper indices
            # For resampled data, we create new sequential indices
            X_res = pd.DataFrame(X_res_array, columns=X_train_scaled.columns)
            y_res = pd.Series(y_res_array, name='class_id')
            
            logger.info(f"Final resampled training set: {X_res.shape}")
            logger.info(f"Class distribution after resampling:\n{y_res.value_counts().sort_index()}")
            
        except Exception as e:
            logger.error(f"Error during SMOTE/Undersampling: {e}")
            logger.warning("Falling back to original training data without resampling")
            X_res, y_res = X_train_scaled, y_train

        # Test set remains unchanged (no augmentation from resampling)
        X_test, y_test = X_test_base_scaled, y_test_base
        logger.info(f"Test set shape: {X_test.shape}")

        # Finalize test set features
        X_test = X_test.fillna(0)
        logger.info(f"Final test set shape: {X_test.shape}")

        # Create datasets and dataloaders
        train_dataset = GeoDataset(X_res.values, y_res.values)
        val_dataset = GeoDataset(X_val_scaled.values, y_val.values)
        test_dataset = GeoDataset(X_test.values, y_test.values)

        # Increased batch size for faster training on GPU
        # Use smaller batch size for T4 GPU (15GB) to avoid OOM errors
        batch_size = 128
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        logger.info(f"Using batch size: {batch_size}")

        # Train the deep learning UNetClassifier (complex architecture with attention, residuals)
        logger.info("Training DeepUNetClassifier...")
        num_classes = len(pd.Series(y_res).unique())
        
        # Create the complex model
        # Enhanced GPU detection and logging
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
        
        logger.info(f"Device: {device}")
        
        # Focus attention on tree vs grass/agriculture (urban is easy to separate)
        tree_grass_indices = None
        band_cols = [c for c in feature_columns[:8] if isinstance(c, str) and ("_mean" in c or c.startswith("mean_"))]
        if len(band_cols) >= 8:
            tree_grass_indices = TREE_GRASS_PRIORITY_BAND_INDICES
            logger.info(
                "Attention focused on tree vs grass/agriculture: upweighting B05, B06, B11, B12 "
                f"(red-edge + SWIR) at indices {tree_grass_indices}."
            )
        # Create model with the complex architecture
        model = UNetClassifier(
            input_size=len(feature_columns),
            num_classes=num_classes,
            tree_grass_band_priority_indices=tree_grass_indices,
        )
        model = model.to(device)
        
        # Default to retrain mode - only train from scratch if explicitly requested
        if not args.retrain:
            logger.info("🔄 DEFAULT MODE: Attempting to load existing model for retraining...")
            args.retrain = True  # Force retrain mode by default
        
        if args.retrain:
            logger.info("🔄 RETRAIN MODE: Loading existing model for retraining...")
            
            # Determine model path
            model_path = local_paths.get('model_path') or (args.model_path if args.model_path else local_paths['model_output'])
            
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"✅ Successfully loaded existing model from: {model_path}")
                    logger.info(f"   Previous best epoch: {checkpoint.get('epoch', 'Unknown')}")
                    logger.info(f"   Previous best F1: {checkpoint.get('val_f1', 'Unknown'):.4f}")
                    
                    # Load optimizer state if available
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("✅ Loaded optimizer state from checkpoint")
                    else:
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        logger.info("🔄 Using new optimizer (no previous state found)")
                    
                    # Normalize and re-save checkpoint to avoid future shape mismatches
                    try:
                        torch.save({
                            'epoch': checkpoint.get('epoch', 0),
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_acc': checkpoint.get('val_acc'),
                            'train_acc': checkpoint.get('train_acc'),
                            'val_loss': checkpoint.get('val_loss'),
                            'train_loss': checkpoint.get('train_loss'),
                            'val_f1': checkpoint.get('val_f1'),
                            'val_precision': checkpoint.get('val_precision'),
                            'val_recall': checkpoint.get('val_recall'),
                            'input_size': len(feature_columns),
                            'num_classes': num_classes
                        }, model_path)
                        logger.info(f"💾 Normalized checkpoint re-saved with input_size={len(feature_columns)} and num_classes={num_classes}")
                    except Exception as save_err:
                        logger.warning(f"Failed to re-save normalized checkpoint: {save_err}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load existing model: {e}")
                    logger.info("🔄 Falling back to training from scratch")
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            else:
                logger.warning(f"Model file not found: {model_path}")
                logger.info("🔄 Falling back to training from scratch")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            logger.info("🆕 TRAIN FROM SCRATCH MODE: Creating new model (explicitly requested)")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Enable debug logging for tensor shapes
        model.debug_shapes = False  # Disabled to reduce logging noise
        
        # Log model architecture
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {total_params:,} parameters, {len(feature_columns)} features → {num_classes} classes")
        
        # Initialize BatchNorm statistics with a forward pass
        logger.info("Initializing BatchNorm...")
        model.train()
        try:
            with torch.no_grad():
                # Get a small batch to initialize BatchNorm
                sample_batch = next(iter(train_loader))[0][:32].to(device)  # Use first 32 samples
                _ = model(sample_batch)
                logger.info("BatchNorm initialized")
        except Exception as e:
            logger.warning(f"BatchNorm init failed: {e}")
            logger.info("Continuing...")
        
        # Train the deep learning model
        # Class weights and label smoothing
        class_counts_train = pd.Series(y_res).value_counts().sort_index()
        num_classes = len(class_counts_train)
        class_weights = (class_counts_train.sum() / (class_counts_train + 1e-8)).values
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

        # Optimizer with weight decay (L2 regularization)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Gradient clipping for training stability
        max_grad_norm = 1.0
        logger.info(f"Using gradient clipping with max_norm={max_grad_norm}")

        # LR scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',          # monitor validation loss
            factor=0.5,          # gentler decay
            patience=5,          # wait longer before reducing
            threshold=1e-4,      # require meaningful improvement
            cooldown=2,          # give a couple of epochs after a reduction
            min_lr=1e-5          # never go below this LR
        )
        
        if args.retrain:
            new_output_path = _next_retrain_suffix_path(PATHS['model_output'])
            logger.info(f"Retrain mode: saving model checkpoint to {new_output_path}")
            PATHS['model_output'] = new_output_path
            local_paths['model_output'] = new_output_path
            os.makedirs(os.path.dirname(local_paths['model_output']), exist_ok=True)

        logger.info("Starting deep learning training...")
        checkpoint_extra = {
            'feature_columns': feature_columns,
            'scaler_mean': getattr(scaler, 'mean_', None),
            'scaler_scale': getattr(scaler, 'scale_', None)
        }
        trained_model = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            scheduler,
            num_epochs=100,
            patience=15,
            checkpoint_extra=checkpoint_extra,
            model_output_path=local_paths['model_output'],
            max_grad_norm=max_grad_norm,
        )
        
        # Safety check: ensure we have a trained model
        if trained_model is None:
            logger.error("Training failed - no model returned. Loading last saved checkpoint...")
            try:
                checkpoint = torch.load(local_paths['model_output'], map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                trained_model = model
                logger.info("Successfully loaded model from checkpoint")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                raise RuntimeError("No trained model available")
        
        # ========================================================================
        # EVALUATE DEEP U-NET MODEL
        # ========================================================================
        logger.info("=" * 60)
        logger.info("EVALUATING DEEP U-NET MODEL")
        logger.info("=" * 60)
        
        trained_model.eval()
        val_pred = []
        val_proba = []
        test_pred = []
        test_proba = []
        
        with torch.no_grad():
            # Validation predictions
            for features, _ in val_loader:
                features = features.to(device)
                outputs = trained_model(features)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                val_pred.extend(predicted.cpu().numpy())
                val_proba.extend(probs.cpu().numpy())
            
            # Test predictions
            for features, _ in test_loader:
                features = features.to(device)
                outputs = trained_model(features)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                test_pred.extend(predicted.cpu().numpy())
                test_proba.extend(probs.cpu().numpy())
        
        val_proba = np.array(val_proba)
        test_proba = np.array(test_proba)
        
        # Deep U-Net metrics
        dunet_val_f1 = f1_score(y_val.values, val_pred, average='macro')
        dunet_test_f1 = f1_score(y_test.values, test_pred, average='macro')
        dunet_val_acc = accuracy_score(y_val.values, val_pred)
        dunet_test_acc = accuracy_score(y_test.values, test_pred)
        
        logger.info(f"Deep U-Net Results:")
        logger.info(f"  Val F1: {dunet_val_f1:.4f}, Acc: {dunet_val_acc:.4f}")
        logger.info(f"  Test F1: {dunet_test_f1:.4f}, Acc: {dunet_test_acc:.4f}")
        
        # --------------------------------------------------------------------
        # Save an additional F1-tagged model checkpoint for easy comparison
        # --------------------------------------------------------------------
        try:
            base_model_path = local_paths.get('model_output', PATHS['model_output'])
            if os.path.exists(base_model_path):
                # Format F1 with 4 decimals and make it filename-safe (e.g. 0.8734 -> f1_0p8734)
                f1_str = f"{dunet_val_f1:.4f}"
                safe_f1 = f1_str.replace(".", "p")
                base_name = os.path.basename(PATHS['model_output'])
                root, ext = os.path.splitext(base_name)
                tagged_name = f"{root}_val{safe_f1}{ext}"
                
                tagged_local = os.path.join(os.path.dirname(base_model_path), tagged_name)
                shutil.copy2(base_model_path, tagged_local)
                logger.info(f"Saved F1-tagged model checkpoint locally to: {tagged_local}")
            else:
                logger.warning(f"Base model checkpoint not found at {base_model_path}; cannot create F1-tagged copy.")
        except Exception as tag_err:
            logger.warning(f"Failed to create F1-tagged model checkpoint: {tag_err}")
        
        # ========================================================================
        # XGBOOST BASELINE BENCHMARKING
        # ========================================================================
        xgb_results = train_xgboost_baseline(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test, y_test, feature_columns
        )
        
        # ========================================================================
        # PROBABILITY CALIBRATION EVALUATION
        # ========================================================================
        logger.info("=" * 60)
        logger.info("EVALUATING PROBABILITY CALIBRATION")
        logger.info("=" * 60)
        
        dunet_calibration = evaluate_calibration(y_test.values, test_proba)
        
        if xgb_results:
            xgb_calibration = evaluate_calibration(y_test.values, xgb_results['test_proba'])
        
        # ========================================================================
        # COMPREHENSIVE EVALUATION REPORT
        # ========================================================================
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE EVALUATION REPORT")
        logger.info("=" * 60)
        
        report_data = {
            'Metric': [
                'F1 (Macro) - Val',
                'F1 (Macro) - Test',
                'Accuracy - Val',
                'Accuracy - Test',
                'Calibration Error (ECE)',
                'Inference Time (ms/sample)',
                'Model Size (MB)'
            ],
            'Deep U-Net (Spatial Split)': [
                f"{dunet_val_f1:.4f}",
                f"{dunet_test_f1:.4f}",
                f"{dunet_val_acc:.4f}",
                f"{dunet_test_acc:.4f}",
                f"{dunet_calibration['mean_ece']:.4f}",
                "High (GPU required)",
                f"{total_params * 4 / (1024**2):.2f}"  # 4 bytes per float32 param
            ]
        }
        
        if xgb_results:
            # Estimate inference time (rough)
            import time as time_module
            start = time_module.time()
            _ = xgb_results['model'].predict(X_test[:100])
            xgb_inference_time = (time_module.time() - start) / 100 * 1000  # ms per sample
            
            report_data['XGBoost Baseline'] = [
                f"{xgb_results['val_f1']:.4f}",
                f"{xgb_results['test_f1']:.4f}",
                f"{xgb_results['val_acc']:.4f}",
                f"{xgb_results['test_acc']:.4f}",
                f"{xgb_calibration['mean_ece']:.4f}",
                f"{xgb_inference_time:.2f}",
                "< 5"
            ]
        
        report_df = pd.DataFrame(report_data)
        logger.info("\n" + report_df.to_string(index=False))
        
        # Save report
        if temp_dir:
            report_path = os.path.join(temp_dir, 'evaluation_report.csv')
            report_df.to_csv(report_path, index=False)
            logger.info(f"\nEvaluation report saved to: {report_path}")
        
        logger.info("=" * 60)
        
        val_pred = np.array(val_pred)

        # Calculate metrics
        accuracy = accuracy_score(y_val, val_pred)
        f1_scores = f1_score(y_val, val_pred, average=None)
        precision_scores = precision_score(y_val, val_pred, average=None)
        recall_scores = recall_score(y_val, val_pred, average=None)

        # Log metrics
        logger.info("\n=== Validation Metrics ===")
        logger.info(f"Overall Accuracy: {accuracy:.4f}")

        # Define the mapping from class IDs to class names
        Subjects = {
            0: "grassland",
            1: "tree",
            2: "urban"
        }

        logger.info("\nPer-Class Metrics:")
        for i, class_name in sorted(Subjects.items()):
            logger.info(f"\n{class_name.upper()}:")
            logger.info(f"F1-Score: {f1_scores[i]:.4f}")
            logger.info(f"Precision: {precision_scores[i]:.4f}")
            logger.info(f"Recall: {recall_scores[i]:.4f}")

        # Capture across-feature attention statistics on test set for interpretability
        logger.info("\nCollecting across-feature attention statistics on test set...")
        # Enable stats for all AcrossFeatureAttention modules
        attn_modules = [m for m in trained_model.modules() if isinstance(m, AcrossFeatureAttention)]
        for m in attn_modules:
            m.collect_stats = True
            m.reset_stats()

        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device)
                _ = trained_model(features)

        # Aggregate and save feature importance from input-level attention and deeper blocks
        feature_names = feature_columns
        model_output_dir = os.path.dirname(local_paths['model_output'])
        os.makedirs(model_output_dir, exist_ok=True)
        csv_path = os.path.join(model_output_dir, 'feature_attention_importance.csv')

        records = []
        for idx, m in enumerate(attn_modules):
            avg_attn = m.get_average_attention()
            if avg_attn is None:
                continue
            # Incoming and outgoing attention per feature
            incoming = avg_attn.mean(axis=0)  # how much others attend to feature j
            outgoing = avg_attn.mean(axis=1)  # how much feature i attends to others
            # Normalize to sum to 1 for readability
            incoming = incoming / (incoming.sum() + 1e-8)
            outgoing = outgoing / (outgoing.sum() + 1e-8)

            # Map to feature names where shapes match input feature count
            if len(feature_names) == len(incoming):
                for f_idx, fname in enumerate(feature_names):
                    records.append({
                        'module_index': idx,
                        'module_type': 'AcrossFeatureAttention',
                        'feature': fname,
                        'incoming_importance': float(incoming[f_idx]),
                        'outgoing_importance': float(outgoing[f_idx])
                    })
            else:
                # If the module is not over raw features, index by generic token id
                for f_idx in range(len(incoming)):
                    records.append({
                        'module_index': idx,
                        'module_type': 'AcrossFeatureAttention',
                        'feature': f'token_{f_idx}',
                        'incoming_importance': float(incoming[f_idx]),
                        'outgoing_importance': float(outgoing[f_idx])
                    })

        if records:
            df_imp = pd.DataFrame(records)
            df_imp.to_csv(csv_path, index=False)
            logger.info(f"Saved attention-based feature importance to {csv_path}")
            # Log top features by incoming importance from the input-level module if present
            input_level = None
            if len(feature_names) > 0:
                input_level = df_imp[df_imp['feature'].isin(feature_names)]
            if input_level is not None and not input_level.empty:
                topk = (input_level.groupby('feature')['incoming_importance']
                        .mean().sort_values(ascending=False).head(10))
                logger.info("Top 10 features by incoming importance (mean across modules):")
                for fname, score in topk.items():
                    logger.info(f"{fname}: {score:.4f}")

        # Evaluate on test set (10% base + undersampling leftovers)
        logger.info("\n=== Test Metrics ===")
        test_pred = []
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device)
                outputs = trained_model(features)
                _, predicted = torch.max(outputs.data, 1)
                test_pred.extend(predicted.cpu().numpy())
        test_pred = np.array(test_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_f1_scores = f1_score(y_test, test_pred, average=None)
        test_precision_scores = precision_score(y_test, test_pred, average=None)
        test_recall_scores = recall_score(y_test, test_pred, average=None)
        logger.info(f"Overall Accuracy: {test_accuracy:.4f}")
        for i, class_name in sorted(Subjects.items()):
            logger.info(f"\n{class_name.upper()}:")
            logger.info(f"F1-Score: {test_f1_scores[i]:.4f}")
            logger.info(f"Precision: {test_precision_scores[i]:.4f}")
            logger.info(f"Recall: {test_recall_scores[i]:.4f}")

        # Generate confusion matrices for validation set
        logger.info("\nGenerating confusion matrices for validation set...")
        confusion_matrix_dir = os.path.join(os.path.dirname(local_paths['model_output']), 'confusion_matrices')
        val_metrics_df = plot_confusion_matrices(y_val, val_pred, list(Subjects.values()), confusion_matrix_dir, dataset_name="Validation")
        logger.info(f"Validation confusion matrices saved to: {confusion_matrix_dir}")
        
        # Generate confusion matrices for test set
        logger.info("\nGenerating confusion matrices for test set...")
        test_metrics_df = plot_confusion_matrices(y_test, test_pred, list(Subjects.values()), confusion_matrix_dir, dataset_name="Test")
        logger.info(f"Test confusion matrices saved to: {confusion_matrix_dir}")

        # Create result DataFrame
        logger.info("Creating result DataFrame...")
        result = gdf.copy()

        # Prepare features for the entire dataset
        logger.info("Preparing features for entire dataset...")
        X_all = result[feature_columns].fillna(0)
        try:
            X_all = pd.DataFrame(scaler.transform(X_all), index=X_all.index, columns=X_all.columns)
        except Exception:
            pass
        logger.info(f"Full dataset shape: {X_all.shape}")

        # Predict on entire dataset using the trained deep learning model
        logger.info("Predicting on full dataset using trained DeepUNetClassifier...")
        trained_model.eval()
        
        # Convert to tensor and predict
        X_all_tensor = torch.FloatTensor(X_all.values).to(device)
        probabilities = []
        predictions = []
        
        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 256  # Use same batch size as training
            for i in range(0, len(X_all_tensor), batch_size):
                batch = X_all_tensor[i:i+batch_size]
                outputs = trained_model(batch)
                batch_probs = torch.softmax(outputs, dim=1)
                batch_preds = torch.argmax(outputs, dim=1)
                
                probabilities.extend(batch_probs.cpu().numpy())
                predictions.extend(batch_preds.cpu().numpy())
        
        probabilities = np.array(probabilities)
        predictions = np.array(predictions)

        # Add predictions to the entire DataFrame
        logger.info("Adding predictions to DataFrame...")
        result['pred_class'] = predictions
        result['confidence'] = probabilities.max(axis=1)

        # Map class names
        result['classification'] = result['pred_class'].map(Subjects)
        result['binary_classification'] = result['classification'].map(lambda x: 'tree' if x == 'tree' else 'no_tree')

        # Rasterize the results
        logger.info("Rasterizing results to GeoTIFF...")

        # Check if geometry column exists
        if 'geometry' not in result.columns or result.geometry.isna().all():
            logger.warning("No geometry column available in results. Skipping rasterization.")
            logger.info("To enable rasterization, ensure the input file contains geometry data.")
        else:
        # Ensure rasterization happens in a projected CRS (meters)
            gdf_for_raster = result
        try:
            if gdf_for_raster.crs is None:
                logger.warning("Input CRS is None; rasterization may fail due to invalid bounds")
            elif not gdf_for_raster.crs.is_projected:
                logger.info("Input CRS is geographic; reprojecting to EPSG:3857 for meter-based rasterization...")
                gdf_for_raster = gdf_for_raster.to_crs(epsg=3857)
        except Exception as e:
            logger.warning(f"CRS check/reprojection failed: {e}. Proceeding with original CRS.")

        # Get bounds and compute pixel size/shape safely
            try:
                bounds = gdf_for_raster.total_bounds  # minx, miny, maxx, maxy
            except AttributeError as e:
                logger.error(f"Cannot get bounds for rasterization: {e}")
                logger.warning("Skipping rasterization due to geometry issues.")
                bounds = None
            
            if bounds is None:
                logger.warning("Bounds are None; skipping rasterization.")
            else:
                resolution = 3  # 3 meters per pixel
                span_x = max(0.0, float(bounds[2] - bounds[0]))
                span_y = max(0.0, float(bounds[3] - bounds[1]))
                width = max(1, int(math.ceil(span_x / float(resolution))))
                height = max(1, int(math.ceil(span_y / float(resolution))))
                transform = from_origin(bounds[0], bounds[3], resolution, resolution)

                # Create multi-class raster
                logger.info(f"Creating multi-class classification raster at {resolution}m resolution...")
                shapes = ((geom, value) for geom, value in zip(gdf_for_raster.geometry, gdf_for_raster['pred_class']))
                raster = rasterio.features.rasterize(
                    shapes=shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )

                # Save multi-class raster
                with rasterio.open(
                    local_paths['raster_output'],
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=np.uint8,
                    crs=result.crs,
                    transform=transform,
                    compress='deflate',
                    zlevel=9,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256
                ) as dst:
                    dst.write(raster, 1)
                    # Add class descriptions
                    dst.write_colormap(1, {
                        0: (0, 255, 0, 255),    # grassland - green
                        1: (0, 128, 0, 255),    # tree - dark green
                        2: (128, 128, 128, 255) # urban - gray
                    })

                # Create binary raster
                logger.info(f"Creating binary classification raster at {resolution}m resolution...")
                binary_shapes = ((geom, 1 if value == 'tree' else 0) for geom, value in zip(gdf_for_raster.geometry, gdf_for_raster['classification']))
                binary_raster = rasterio.features.rasterize(
                    shapes=binary_shapes,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )

                # Save binary raster
                with rasterio.open(
                    local_paths['binary_raster_output'],
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=np.uint8,
                    crs=result.crs,
                    transform=transform,
                    compress='deflate',
                    zlevel=9,
                    tiled=True,
                    blockxsize=256,
                    blockysize=256
                ) as dst:
                    dst.write(binary_raster, 1)
                    # Add binary class descriptions
                    dst.write_colormap(1, {
                        0: (0, 255, 0, 255),    # no_tree - green
                        1: (0, 128, 0, 255)     # tree - dark green
                    })

                logger.info("Rasterization completed successfully.")

        try:
            logger.info("\n=== Saved Files ===")
            logger.info(f"PyTorch model: {PATHS['model_output']}")
            if 'geometry' in result.columns and not result.geometry.isna().all():
                logger.info(f"Multi-class raster: {PATHS['raster_output']}")
                logger.info(f"Binary raster: {PATHS['binary_raster_output']}")
                logger.info("=" * 30)
        except Exception as e:
            logger.error(f"Failed to save/upload files: {e}")
            raise
    finally:
        # Clean up temporary files
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                logger.info("Temporary files cleaned up successfully")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")

if __name__ == '__main__':
    main()