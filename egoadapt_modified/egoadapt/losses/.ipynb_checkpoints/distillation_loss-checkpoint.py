import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class DistillWeights:
    alpha: float = 0.5
    beta: float = 0.1
    T: float = 2.0

def kd_loss(student_logits, teacher_logits, T=2.0):
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction='batchmean') * (T * T)

def groundtruth_loss(student_logits, labels):
    return F.cross_entropy(student_logits, labels)

def l1_feature_loss(student_logits, teacher_logits):
    ps, pt = F.softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1)
    return F.l1_loss(ps, pt)

def distillation_loss(student_logits, teacher_logits, gt_labels, w: DistillWeights):
    L_KD = kd_loss(student_logits, teacher_logits, T=w.T)
    L_GT = groundtruth_loss(student_logits, gt_labels)
    L_1  = l1_feature_loss(student_logits, teacher_logits)
    L_phi = w.alpha * L_KD + (1 - w.alpha) * L_GT + w.beta * L_1
    return dict(L_KD=L_KD, L_GT=L_GT, L_1=L_1, L_phi=L_phi)