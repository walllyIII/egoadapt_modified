import torch
import torch.nn.functional as F

def policy_loss(logits, targets, gates_hard, lambdas, gamma_miscls=1.0, power=2.0):
    """
    Implements:  E_{(M,y)~D_train} [ -y log P(M; Θ) + Σ_k λ_k C_k ]
    where C_k = (fraction of frames kept by modality k)^power
    """
    L_cls = F.cross_entropy(logits, targets)
    B, T, _ = gates_hard[0].shape
    total = float(B * T)
    cost = 0.0
    for k, gk in enumerate(gates_hard):
        frac = gk.sum() / (total + 1e-8)
        cost += lambdas[k] * (frac ** power)
    L_pi = gamma_miscls * L_cls + cost
    return L_cls, cost, L_pi