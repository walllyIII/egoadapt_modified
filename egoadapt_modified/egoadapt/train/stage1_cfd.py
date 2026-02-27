# egoadapt/train/stage1_cfd.py
from __future__ import annotations
import torch
from egoadapt.models.fusion import CrossModalStudentPhi
from egoadapt.losses.distillation_loss import distillation_loss, DistillWeights

@torch.no_grad()
def forward_teacher(teacher, batch):
    return teacher(**batch["teacher_inputs"])  # adapt key names per teacher


def train_step(model_phi: CrossModalStudentPhi,
               teacher,
               batch,
               w: DistillWeights,
               opt):
    model_phi.train()
    out = model_phi(batch["I"], batch["A"], batch["B"])  # logits in out["logits"]
    t_logits = forward_teacher(teacher, batch)
    losses = distillation_loss(out, t_logits, batch["y"], w)
    opt.zero_grad(set_to_none=True)
    losses["L_phi"].backward()
    opt.step()
    return {k: float(v.detach().cpu()) for k, v in losses.items()}