import torch
from torch.utils.data import DataLoader
from egoadapt.data.datasets import EasyComDataset
from egoadapt.models.fusion import CrossModalStudentPhi
import torch.nn.functional as F

# ---------------------------
# 1. Dataset + Dataloader
# ---------------------------
PROCESSED_ROOT = "/root/autodl-tmp/egoadapt/data/easycom_processed"
CSV_PATH = "/root/autodl-tmp/egoadapt/data/easycom_processed/test.csv"  # 测试集csv

dataset = EasyComDataset(root=PROCESSED_ROOT, csv_list=CSV_PATH)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)  # 可调整 batch

# ---------------------------
# 2. 加载模型
# ---------------------------
phi = CrossModalStudentPhi(n_classes=64, d=256)  # n_classes 按实际改
checkpoint_path = "/root/autodl-tmp/egoadapt/checkpoints/best_phi.pth"
phi.load_state_dict(torch.load(checkpoint_path))
phi.eval()
phi = phi.cuda()  # GPU加速

# ---------------------------
# 3. 辅助函数：对齐和重塑
# ---------------------------
def align_audio_time_steps(A_seq, target_T=16):
    B, curr_T, F = A_seq.shape
    if curr_T > target_T:
        return A_seq[:, :target_T, :]
    elif curr_T < target_T:
        pad_size = (0, 0, 0, target_T - curr_T, 0, 0)
        return F.pad(A_seq, pad_size, mode='constant', value=0)
    else:
        return A_seq

def reshape_vision_input(I_seq, target_H=32, target_W=32):
    B, T, D = I_seq.shape
    target_dim = 3 * target_H * target_W
    I_flat = I_seq.reshape(B, T*D)
    if T*D < target_dim:
        I_flat = F.pad(I_flat, (0, target_dim - T*D))
    elif T*D > target_dim:
        I_flat = I_flat[:, :target_dim]
    I_img = I_flat.reshape(B, 3, target_H, target_W)
    return I_img

def reshape_audio_input(A_seq):
    return A_seq.unsqueeze(1).permute(0,1,3,2)  # [B,1,F,T]

def reshape_behavior_input(B_seq):
    B, T, C = B_seq.shape
    B_flat = B_seq.reshape(B, C*T)
    B_1d = F.pad(B_flat, (0, 12*16 - C*T)).reshape(B, 12, 16)
    return B_1d

# ---------------------------
# 4. 推理整个测试集
# ---------------------------
all_preds = []
all_labels = []

if __name__ == "__main__":
    with torch.no_grad():
        for batch_seq in dataloader:
            # 对齐 & 重塑
            I_seq = reshape_vision_input(batch_seq["I_seq"]).cuda()
            A_seq = align_audio_time_steps(batch_seq["A_seq"])
            A_seq = reshape_audio_input(A_seq).cuda()
            B_seq = reshape_behavior_input(batch_seq["B_seq"]).cuda()
    
            # Forward
            out = phi(I_seq, A_seq, B_seq)
            pred = out["logits"].argmax(-1)
    
            all_preds.append(pred.cpu())
            all_labels.append(batch_seq["y_asl"])
    
    # ---------------------------
    # 5. 拼接结果
    # ---------------------------
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # ---------------------------
    # 6. 计算准确率
    # ---------------------------
    accuracy = (all_preds == all_labels).float().mean()
    print("Test accuracy:", accuracy.item())
    
    # ---------------------------
    # 7. 保存预测结果
    # ---------------------------
    torch.save({"preds": all_preds, "labels": all_labels}, "asl_predictions.pt")
    print("Predictions saved to asl_predictions.pt")