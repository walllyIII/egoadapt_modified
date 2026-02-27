# egoadapt/models/fusion.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import TinyVisionEncoder, TinyAudioEncoder, TinyBehaviorEncoder

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers=2, p_drop=0.0):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, d_hidden), nn.ReLU(inplace=True), nn.Dropout(p_drop)]
            d = d_hidden
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class FusionHead(nn.Module):
    """Late fusion with learnable modality weights, shared across tasks (phi)."""
    def __init__(self, d: int, n_classes: int):
        super().__init__()
        self.alpha_v = nn.Parameter(torch.tensor(1.0))
        self.alpha_a = nn.Parameter(torch.tensor(1.0))
        self.alpha_b = nn.Parameter(torch.tensor(1.0))
        self.fuse = MLP(3 * d, 2 * d, d, n_layers=2, p_drop=0.1)
        self.cls = nn.Linear(d, n_classes)
    def forward(self, z_v, z_a, z_b):
        z = torch.cat([self.alpha_v * z_v, self.alpha_a * z_a, self.alpha_b * z_b], dim=-1)
        z = self.fuse(z)
        logits = self.cls(z)
        return z, logits

class CrossModalStudentPhi(nn.Module):
    """phi: student that fuses (vision, audio, behavior) and predicts class logits."""
    def __init__(self, n_classes: int, d=256, audio_ch=1, beh_dim=12):
        super().__init__()
        self.enc_v = TinyVisionEncoder(d=d)
        self.enc_a = TinyAudioEncoder(in_ch=audio_ch, d=d)
        self.enc_b = TinyBehaviorEncoder(in_ch=beh_dim, d=d)
        self.head = FusionHead(d=d, n_classes=n_classes)

    def forward(self, I, A, B):
        zI = self.enc_v(I)
        zA = self.enc_a(A)
        zB = self.enc_b(B)
        z_phi, logits = self.head(zI, zA, zB)
        return dict(zI=zI, zA=zA, zB=zB, z_phi=z_phi, logits=logits)