# egoadapt/models/encoders.py
"""
Encoders for EgoAdapt.

- Activity Recognition (AR): FasterNetLite (small) student inspired by FasterNet.
- Behavior Anticipation (BA) & Active Speaker Localization (ASL): lightweight encoders:
    * Video: 2x (Conv2d → BN → ReLU → MaxPool) + GAP + FC
    * Audio: 2x (Conv2d → BN → ReLU → MaxPool) + GAP + FC  (expects spectrogram [B,1,F,T])
    * Sensor: 2x (Conv1d → BN → ReLU → MaxPool) + GAP + FC (expects [B,C,T])

Factory: build_encoder(task, modality, d)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------------------

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, s, p),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------
# Lightweight encoders for BA / ASL
# ---------------------------------------------------------------------

class TinyVisionEncoder(nn.Module):
    """Two 2D conv blocks + GAP + FC. Input [B,3,H,W] -> [B,d]."""
    def __init__(self, in_ch: int = 3, d: int = 128):
        super().__init__()
        self.conv1 = ConvBlock2D(in_ch, 32)
        self.conv2 = ConvBlock2D(32, 64)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.head(x)


class TinyAudioEncoder(nn.Module):
    """Two 2D conv blocks + GAP + FC for spectrograms. Input [B,1,F,T] -> [B,d]."""
    def __init__(self, in_ch: int = 1, d: int = 128):
        super().__init__()
        self.conv1 = ConvBlock2D(in_ch, 32)
        self.conv2 = ConvBlock2D(32, 64)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.head(x)


class TinyBehaviorEncoder(nn.Module):
    """Two 1D conv blocks + GAP + FC. Input [B,C,T] -> [B,d]."""
    def __init__(self, in_ch: int = 12, d: int = 128):
        super().__init__()
        self.conv1 = ConvBlock1D(in_ch, 64)
        self.conv2 = ConvBlock1D(64, 128)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.head(x)


# ---------------------------------------------------------------------
# FasterNet-inspired student for Activity Recognition
# ---------------------------------------------------------------------

class PConv(nn.Module):
    """
    Partial Conv (PConv) as described in FasterNet: process only a fraction of channels
    with a depthwise 3x3 while bypassing the rest, then expand with 1x1.
    """
    def __init__(self, in_ch: int, out_ch: int, ratio: float = 0.25):
        super().__init__()
        self.active_channels = max(1, int(in_ch * ratio))
        self.dw = nn.Conv2d(self.active_channels, self.active_channels, kernel_size=3, padding=1,
                            groups=self.active_channels, bias=False)
        self.bn = nn.BatchNorm2d(self.active_channels)
        self.relu = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_act = self.active_channels
        x1, x2 = x.split([c_act, x.shape[1] - c_act], dim=1) if x.shape[1] > c_act else (x, torch.zeros_like(x))
        y1 = self.relu(self.bn(self.dw(x1)))
        y = torch.cat([y1, x2], dim=1) if x2.numel() > 0 else y1
        return self.expand(y)


class FasterBlock(nn.Module):
    """PConv -> PWConv(1x1) + BN + ReLU; residual connection when shapes match."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pconv = PConv(in_ch, out_ch)
        self.pw = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.use_res = (in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pconv(x)
        y = self.pw(y)
        return x + y if self.use_res and x.shape == y.shape else y


class FasterNetLite(nn.Module):
    """
    A compact 4-stage FasterNet-style backbone for AR.
    Input [B,3,H,W] -> [B,d].
    """
    def __init__(self, in_ch: int = 3, d: int = 256):
        super().__init__()
        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            FasterBlock(32, 32),
        )
        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            FasterBlock(64, 64),
        )
        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            FasterBlock(128, 128),
        )
        # Stage 4
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            FasterBlock(256, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def build_encoder(task: str, modality: str, d: int = 256) -> nn.Module:
    """
    Args:
        task: 'activity_recognition' | 'behaviour_anticipation' | 'active_speaker_localization'
        modality: 'video' | 'audio' | 'sensor'
        d: output embedding dimension
    """
    task = task.lower()
    modality = modality.lower()

    if task == "activity_recognition":
        if modality != "video":
            raise ValueError("For AR, FasterNetLite expects video modality.")
        return FasterNetLite(in_ch=3, d=d)

    if task in ("behaviour_anticipation", "behavior_anticipation", "active_speaker_localization"):
        if modality == "video":
            return TinyVisionEncoder(in_ch=3, d=d)
        if modality == "audio":
            return TinyAudioEncoder(in_ch=1, d=d)
        if modality == "sensor":
            return TinyBehaviorEncoder(in_ch=12, d=d)
        raise ValueError(f"Unknown modality for BA/ASL: {modality}")

    raise ValueError(f"Unknown task: {task}")
