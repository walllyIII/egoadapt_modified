import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from egoadapt.models.fusion import CrossModalStudentPhi
from egoadapt.teachers.swl_teacher_lite import SWLTeacherLite
from egoadapt.losses.distillation_loss import DistillWeights
from egoadapt.train.stage1_cfd import train_step
from egoadapt.data.datasets import EasyComDataset

from egoadapt.models.policy_pi_avloc_ba import PolicyNetASL_BA
from egoadapt.train.stage2_policy_avloc_ba import train_step_policy_avloc_ba

from egoadapt.train.stage3_joint import train_step_joint
from egoadapt.losses.distillation_loss import DistillWeights

from egoadapt.utils.optim import make_opts

PROCESSED_ROOT="/root/autodl-tmp/egoadapt/data/easycom_processed"
CSV_PATH="/root/autodl-tmp/egoadapt/data/easycom_processed/train.csv"

dataset = EasyComDataset(root=PROCESSED_ROOT, csv_list=CSV_PATH)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

def align_audio_time_steps(A_seq, target_T=16):
    """
    将音频特征的时间步从49对齐到16
    :param A_seq: 输入音频特征，shape [32,49,768]
    :param target_T: 目标时间步，默认16
    :return: 对齐后的音频特征，shape [32,16,768]
    """
    batch_size, curr_T, feat_dim = A_seq.shape
    
    # 情况1：当前时间步 > 目标时间步 → 截断前16个时间步
    if curr_T > target_T:
        A_seq_aligned = A_seq[:, :target_T, :]  # 取前16步，shape [32,16,768]
    
    # 情况2：当前时间步 < 目标时间步 → 补0填充到16步
    elif curr_T < target_T:
        pad_size = (0, 0, 0, target_T - curr_T, 0, 0)  # (左,右,上,下,前,后) 对应最后三维
        A_seq_aligned = torch.nn.functional.pad(A_seq, pad_size, mode='constant', value=0)
    
    # 情况3：时间步正好等于16 → 直接返回
    else:
        A_seq_aligned = A_seq
    
    return A_seq_aligned

def reshape_vision_input(I_seq, target_H=128, target_W=64):
    """
    将视频序列特征 [B,16,1000] → 图像格式 [B*16,3,H,W]
    :param I_seq: [B, T, D] 视频序列特征（T=16, D=1000）
    :param target_H/W: 目标图像尺寸
    :return: [B*16, 3, H, W] 图像张量
    """
    B, T, D = I_seq.shape
    # 1. 展平为 [B, T*D]
    I_flat = I_seq.reshape(B, T * D)
    # 2. 补0/截断到 3*H*W（适配3通道图像）
    target_dim = 3 * target_H * target_W
    if T*D < target_dim:
        I_flat = F.pad(I_flat, (0, target_dim - T*D), mode="constant", value=0)
    elif T*D > target_dim:
        I_flat = I_flat[:, :target_dim]
    # 3. 重塑为 [B*T, 3, H, W]
    I_img = I_flat.reshape(B, 3, target_H, target_W)
    return I_img

def reshape_audio_input(A_seq):
    """
    将音频序列特征 [B,16,768] → 频谱格式 [B,1,F,T]（F=768, T=16）
    :param A_seq: [B, T, F] 音频序列特征（T=16, F=768）
    :return: [B, 1, F, T] 频谱张量
    """
    # [B,16,768] → 加通道维 [B,1,16,768] → 转置为 [B,1,768,16]（F=768, T=16）
    A_spec = A_seq.unsqueeze(1).permute(0, 1, 3, 2)
    return A_spec

def reshape_behavior_input(B_seq):
    """
    将行为序列特征 [B,16,3] → 1D卷积格式 [B,C,T]（C=3, T=16）
    :param B_seq: [B, T, C] 行为序列特征（T=16, C=3）
    :return: [B, C, T] 1D序列张量
    """
    # [B,16,3] → 转置为 [B,3,16]
    B,C,T=B_seq.shape
    B_flat=B_seq.reshape(B,C*T)
    target_dim = 12 * 16
    if T*C < target_dim:
        B_flat = F.pad(B_flat, (0, target_dim - T*C), mode="constant", value=0)
    elif T*C > target_dim:
        B_flat = B_flat[:, :target_dim]
    B_1d = B_flat.reshape(B, 12, 16)
    return B_1d

best_loss = float('inf')  # 初始化最优loss
save_path = "/root/autodl-tmp/egoadapt/checkpoints"
os.makedirs(save_path, exist_ok=True)

for epoch in range(500):
    print(f"epoch :{epoch}")
    for batch_seq_raw in dataloader:
        
        batch_seq = batch_seq_raw
        
        batch_seq['A_seq']=align_audio_time_steps(batch_seq['A_seq'])
        I_seq = batch_seq["I_seq"]
        A_seq = batch_seq["A_seq"]
        B_seq = batch_seq["B_seq"]
    
        I_seq_raw=I_seq
        A_seq_raw=A_seq
        B_seq_raw=B_seq
        
        batch_seq["I_seq"] = reshape_vision_input(I_seq)
        batch_seq["A_seq"] = reshape_audio_input(A_seq)
        batch_seq["B_seq"] = reshape_behavior_input(B_seq)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        batch_seq["I_seq"] = batch_seq["I_seq"].to(device)
        batch_seq["A_seq"] = batch_seq["A_seq"].to(device)
        batch_seq["B_seq"] = batch_seq["B_seq"].to(device)
        batch_seq["y_asl"] = batch_seq["y_asl"].to(device)
        I_seq_raw = I_seq_raw.to(device)
        A_seq_raw = A_seq_raw.to(device)
        B_seq_raw = B_seq_raw.to(device)
        
        phi = CrossModalStudentPhi(n_classes=64, d=256)
        teacher = SWLTeacherLite(d=256, n_classes_asl=64, n_classes_ba=20)
        
        phi = phi.to(device)
        teacher = teacher.to(device)
        
        print(f"I_seq: {I_seq_raw.shape}")
        print(f"A_seq: {A_seq_raw.shape}")
        print(f"B_seq: {B_seq_raw.shape}")
        
        with torch.no_grad():
            teacher_outputs = teacher(I_seq_raw[:,:,:256], A_seq_raw[:,:,:256], B_seq_raw)
            teacher_logits_seq = teacher_outputs["asl_logits"]
        print(type(teacher_logits_seq))
        print(teacher_logits_seq.shape)

        teacher_logits_seq=teacher_logits_seq.to(device)
        
        pi_ab = PolicyNetASL_BA(d_feat=256, n_modalities=3, audio_channels=1)
        _,opt_policy,opt_joint = make_opts(phi, pi_ab)

        #opt_policy=opt_policy.to(device)
        #opt_joint=opt_joint.to(device)
        pi_ab = pi_ab.to(device)
        
        # Stage 2 policy step
        logs_policy = train_step_policy_avloc_ba(phi, pi_ab, batch_seq, lambdas=[0.3,0.3,0.3], task="asl", opt=opt_policy)
        print(logs_policy)
    
        # Stage 3 joint step
        logs_joint = train_step_joint(
            model_phi=phi,
            model_pi=pi_ab,
            batch_seq=batch_seq,
            teacher_logits_seq=teacher_logits_seq,
            cfd_w=DistillWeights(alpha=0.5,beta=0.1,T=2.0),
            lambdas=[0.3,0.3,0.3],
            gamma_miscls=1.0,
            tau=0.8,
            eta1=1.0,
            eta2=1.0,
            opt=opt_joint,
        )
        print(logs_joint)
        
        # ---------------------------
        # 保存最优模型（以joint loss为准）
        # ---------------------------
        current_loss = logs_joint.get('L_theta', None)
        
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(phi.state_dict(), os.path.join(save_path, "best_phi.pth"))
            torch.save(pi_ab.state_dict(), os.path.join(save_path, "best_pi_ab.pth"))
            print(f"Saved new best model! Loss: {best_loss}")
