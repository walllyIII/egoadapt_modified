# egoadapt/models/teachers/tim_teacher.py
"""
TIM (CVPR'24) teacher wrapper for Action Recognition.

- Requires the official repo available in your PYTHONPATH or as a submodule:
    git submodule add https://github.com/JacobChalk/TIM external/TIM
    pip install -e external/TIM

- TIM expects pre-extracted features (see their feature_extractors/). This wrapper
  takes video/audio features or raw clips (if you wrap extraction yourself) and
  returns logits for distillation.
"""
from __future__ import annotations
import torch
import torch.nn as nn

try:
    # Typical structure (subject to change if the repo updates)
    # You may need to adjust imports to match the exact filenames in recognition/
    from TIM.recognition.model import RecognitionModel  # type: ignore
except Exception as e:
    RecognitionModel = None

class TIMTeacher(nn.Module):
    def __init__(self, cfg: dict, ckpt_path: str | None = None, device: str = "cpu"):
        super().__init__()
        if RecognitionModel is None:
            raise ImportError("TIM repo not found. Install it and ensure it's on PYTHONPATH.")
        self.model = RecognitionModel(cfg)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location=device)
            # handle different checkpoint keys (e.g., 'state_dict' vs raw)
            state = sd.get("state_dict", sd)
            self.model.load_state_dict(state, strict=False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, vid_feats: torch.Tensor, aud_feats: torch.Tensor, interval_queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vid_feats:  [B, T_v, D_v] pre-extracted video features (per TIM instructions)
            aud_feats:  [B, T_a, D_a] pre-extracted audio features
            interval_queries: [B, Q, 2] (start, end) or similar structure as TIM expects
        Returns:
            logits: [B, C] action class logits
        """
        # The exact call signature depends on TIM's RecognitionModel; adapt if needed
        out = self.model(vid_feats=vid_feats, aud_feats=aud_feats, queries=interval_queries)
        logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
        return logits
