import os
import torch
import cv2
import chardet
from torchvision.models import vit_b_16
from torchvision.transforms import Compose, ToTensor, Normalize
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from egoadapt.data.datasets import EasyComDataset
from torch.utils.data import DataLoader

def load_json_utf8(trans_path):
    """
    读取任意编码的JSON文件，并转换为UTF-8编码后解析
    :param trans_path: JSON文件路径
    :return: 解析后的JSON数据（dict/list）
    """
    # 步骤1：以二进制模式读取文件（避免解码错误）
    with open(trans_path, 'rb') as f:
        raw_bytes = f.read()
    
    # 步骤2：检测文件原始编码
    detected = chardet.detect(raw_bytes)
    src_encoding = detected['encoding'] or 'gbk'  # 兜底编码
    
    # 步骤3：将原始编码转为UTF-8字符串
    try:
        # 用原始编码解码字节 → 字符串
        json_str = raw_bytes.decode(src_encoding)
    except UnicodeDecodeError:
        # 兜底：用ISO-8859-1解码（兼容所有字节）
        json_str = raw_bytes.decode('iso-8859-1')
    
    # 步骤4：解析UTF-8格式的JSON字符串
    trans_data = json.loads(json_str)
    return trans_data

# from video to frames
# 配置 autodl-tmp/egoadapt/data/EasyComDataset
RAW_VIDEO_ROOT = "/root/autodl-tmp/egoadapt/data/EasyComDataset/Main/Video_Compressed"
# 原始视频根目录
PROCESSED_FRAMES_ROOT = "/root/autodl-tmp/egoadapt/data/easycom_processed/frames"  # 处理后帧特征目录
T = 16  # 序列长度（固定取16帧）
D_i = 768  # ViT-B/16的特征维度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化预训练特征提取模型
vit_model = vit_b_16(pretrained=True).eval().to(DEVICE)
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
'''
# 创建输出目录
os.makedirs(PROCESSED_FRAMES_ROOT, exist_ok=True)

# 遍历所有session和视频文件
for session_dir in os.listdir(RAW_VIDEO_ROOT):
    session_path = os.path.join(RAW_VIDEO_ROOT, session_dir)
    if not os.path.isdir(session_path):
        continue
    session_num = session_dir.replace("Session_", "")  # 提取session编号
    
    for video_file in os.listdir(session_path):
        if not video_file.endswith(".mp4"):
            continue
        file_prefix = video_file[:-4]  # 如 "01-02-003"
        sample_id = f"session_{session_num}_{file_prefix}"  # 样本ID
        video_path = os.path.join(session_path, video_file)
        
        # 1. 读取视频并抽帧
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while cap.isOpened() and frame_count < T:
            ret, frame = cap.read()
            if not ret:
                break
            # 转换为RGB + 调整尺寸（ViT输入224x224）
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(transform(frame))
            frame_count += 1
        cap.release()
        
        # 2. 统一序列长度T（不足补零）
        frames = torch.stack(frames) if frames else torch.empty(0, 3, 224, 224)
        if len(frames) < T:
            pad = torch.zeros(T - len(frames), 3, 224, 224)
            frames = torch.cat([frames, pad], dim=0)
        else:
            frames = frames[:T]
        
        # 3. 提取特征 [T, 768]
        with torch.no_grad():
            frames = frames.to(DEVICE)
            feat = vit_model(frames)  # 取CLS token
        
        # 4. 保存为.pt文件
        torch.save(feat.cpu(), os.path.join(PROCESSED_FRAMES_ROOT, f"{sample_id}.pt"))
        print(f"Processed video: {sample_id}")
'''
print("videos is down")

# from audio to audio
# 配置
RAW_AUDIO_ROOT = "/root/autodl-tmp/egoadapt/data/EasyComDataset/Main/Glasses_Microphone_Array_Audio"  # AR眼镜音频根目录
PROCESSED_AUDIO_ROOT = "/root/autodl-tmp/egoadapt/data/easycom_processed/audio"
D_a = 768  # Wav2Vec2特征维度
SAMPLE_RATE = 16000

# 初始化Wav2Vec2模型
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval().to(DEVICE)
'''
os.makedirs(PROCESSED_AUDIO_ROOT, exist_ok=True)

# 遍历音频文件
for session_dir in os.listdir(RAW_AUDIO_ROOT):
    session_path = os.path.join(RAW_AUDIO_ROOT, session_dir)
    if not os.path.isdir(session_path):
        continue
    session_num = session_dir.replace("Session_", "")
    
    for audio_file in os.listdir(session_path):
        if not audio_file.endswith(".wav"):
            continue
        file_prefix = audio_file[:-4]
        sample_id = f"session_{session_num}_{file_prefix}"
        audio_path = os.path.join(session_path, audio_file)
        
        # 1. 读取音频（6通道转单通道，取第一通道）
        wav, sr = torchaudio.load(audio_path)
        wav = wav[0:1, :]  # 取第一通道
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
        
        # 2. 对齐视频长度：计算需要的音频样本数（T帧 × 每帧音频样本数）
        # 假设视频帧率=16fps，音频16kHz → 每帧对应1000个音频样本
        audio_samples_needed = T * 1000
        if wav.shape[1] < audio_samples_needed:
            pad = torch.zeros(1, audio_samples_needed - wav.shape[1])
            wav = torch.cat([wav, pad], dim=1)
        else:
            wav = wav[:, :audio_samples_needed]
        
        # 3. 提取音频特征
        inputs = feature_extractor(wav.squeeze().cpu().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs["input_values"].to(DEVICE)
        with torch.no_grad():
            audio_feat = wav2vec_model(input_values).last_hidden_state  # [1, T, 768]
        
        # 4. 保存特征 [T, 768]
        torch.save(audio_feat.squeeze(0).cpu(), os.path.join(PROCESSED_AUDIO_ROOT, f"{sample_id}.pt"))
        print(f"Processed audio: {sample_id}")
'''

print("audios is down")

# from tracked_poses to gaze
# 配置
RAW_POSE_ROOT = "/root/autodl-tmp/egoadapt/data/EasyComDataset/Main/Tracked_Poses"
PROCESSED_GAZE_ROOT = "/root/autodl-tmp/egoadapt/data/easycom_processed/gaze"
'''
os.makedirs(PROCESSED_GAZE_ROOT, exist_ok=True)

def quat_to_dir_vector(quat):
    """四元数转头部朝向单位向量（w,x,y,z格式）"""
    rot = R.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]])
    # 初始朝向（前向）向量
    forward = np.array([0, 0, 1])
    # 旋转后的向量
    dir_vec = rot.apply(forward)
    # 归一化为单位向量
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    return dir_vec

# 遍历姿态文件
for session_dir in os.listdir(RAW_POSE_ROOT):
    session_path = os.path.join(RAW_POSE_ROOT, session_dir)
    if not os.path.isdir(session_path):
        continue
    session_num = session_dir.replace("Session_", "")
    
    for pose_file in os.listdir(session_path):
        if not pose_file.endswith(".json"):
            continue
        file_prefix = pose_file[:-5]
        sample_id = f"session_{session_num}_{file_prefix}"
        pose_path = os.path.join(session_path, pose_file)
        
        # 1. 解析JSON
        with open(pose_path, "r") as f:
            pose_data = json.load(f)
        
        # 2. 提取AR眼镜佩戴者的头部姿态（假设ID=0，需根据metadata确认）
        gaze_vectors = []
        
        print(type(pose_data))

        frame_list = pose_data if isinstance(pose_data, list) else pose_data.get("frames", [])
            
        # 取前T帧
        for frame_data in frame_list[:T]:
            dir_vec = np.zeros(3)
            # 遍历参与者，找到AR眼镜佩戴者（假设ID=0，可根据实际调整）
            for participant in frame_data.get("participants", []):
                if participant.get("id") == 0:  # AR佩戴者ID
                    quat = participant.get("rotation", {})
                    dir_vec = quat_to_dir_vector(quat)
                    break
            gaze_vectors.append(dir_vec)
        
        # 3. 统一序列长度T
        gaze_vectors = np.array(gaze_vectors)
        if len(gaze_vectors) < T:
            pad = np.zeros((T - len(gaze_vectors), 3))
            gaze_vectors = np.vstack([gaze_vectors, pad])
        else:
            gaze_vectors = gaze_vectors[:T]
        
        # 4. 转换为tensor并保存
        gaze_tensor = torch.tensor(gaze_vectors, dtype=torch.float32)
        torch.save(gaze_tensor, os.path.join(PROCESSED_GAZE_ROOT, f"{sample_id}.pt"))
        print(f"Processed gaze: {sample_id}")
'''
print("poses is down")


# prepare labels 
RAW_TRANSCRIBE_ROOT = "/root/autodl-tmp/egoadapt/data/EasyComDataset/Main/Speech_Transcriptions"
PROCESSED_LABELS_ROOT = "/root/autodl-tmp/egoadapt/data/easycom_processed/labels"

os.makedirs(PROCESSED_LABELS_ROOT, exist_ok=True)

# 遍历转录文件
for session_dir in os.listdir(RAW_TRANSCRIBE_ROOT):
    session_path = os.path.join(RAW_TRANSCRIBE_ROOT, session_dir)
    if not os.path.isdir(session_path):
        continue
    session_num = session_dir.replace("Session_", "")
    
    for trans_file in os.listdir(session_path):
        if not trans_file.endswith(".json"):
            continue
        file_prefix = trans_file[:-5]
        sample_id = f"session_{session_num}_{file_prefix}"
        trans_path = os.path.join(session_path, trans_file)
        
        # 1. 解析JSON
        # with open(trans_path, "r") as f:
        #     trans_data = json.load(f)
        trans_data = load_json_utf8(trans_path)
        
        # 2. 判断是否有主动说话人（示例规则：存在目标说话人语音则标签=1）
        asl_label = 0
        for speech in trans_data:
            #print(speech["Target_of_Speech"])
            if speech["Target_of_Speech"] == "[2]":  # 说话目标是AR佩戴者
                asl_label = 1
                break
        
        # 3. 保存标签（LongTensor）
        label_tensor = torch.tensor(asl_label, dtype=torch.long)
        torch.save(label_tensor, os.path.join(PROCESSED_LABELS_ROOT, f"{sample_id}.pt"))
        print(f"Processed label: {sample_id} -> {asl_label}")

print("labels is down")

# prepare csv
PROCESSED_ROOT = "/root/autodl-tmp/egoadapt/data/easycom_processed"
CSV_PATH = os.path.join(PROCESSED_ROOT, "train.csv")

# 获取所有样本ID（从frames文件夹提取）
sample_ids = [f[:-3] for f in os.listdir(os.path.join(PROCESSED_ROOT, "frames")) if f.endswith(".pt")]

# 保存为CSV
with open(CSV_PATH, "w") as f:
    for sample_id in sample_ids:
        f.write(f"{sample_id}\n")

print(f"Generated CSV with {len(sample_ids)} samples")



# check the processed dataset
dataset = EasyComDataset(root=PROCESSED_ROOT, csv_list=CSV_PATH)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 验证第一个批次
for batch_seq in dataloader:
    print("=== 验证batch_seq维度 ===")
    print(f"I_seq: {batch_seq['I_seq'].shape}")  # [32, 16, 768]
    print(f"A_seq: {batch_seq['A_seq'].shape}")  # [32, 16, 768]
    print(f"B_seq: {batch_seq['B_seq'].shape}")  # [32, 16, 3]
    print(f"y_asl: {batch_seq['y_asl'].shape}")  # [32]
    break