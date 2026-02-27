import torch
import torchaudio
import os
from torch.utils.data import Dataset
from torchvision.io import read_video

class EPICKitchensDataset(Dataset):
    """Action Recognition dataset loader for TIM teacher and phi student."""
    def __init__(self, root, csv_list, clip_len=16, audio_len=16000):
        self.root = root
        self.samples = [l.strip() for l in open(csv_list)]
        self.clip_len = clip_len
        self.audio_len = audio_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_id = self.samples[idx]
        video_path = os.path.join(self.root, 'videos', f'{vid_id}.mp4')
        audio_path = os.path.join(self.root, 'audio', f'{vid_id}.wav')
        beh_path = os.path.join(self.root, 'behavior', f'{vid_id}.pt')

        # Video: [T, H, W, C] -> [C, T, H, W]
        video, _, _ = read_video(video_path)
        video = video.permute(3, 0, 1, 2) / 255.0
        video = video[:, :self.clip_len]

        # Audio waveform
        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        wav = wav[..., :self.audio_len]

        # Behavior features (sensor or skeleton)
        behavior = torch.load(beh_path)

        label = int(vid_id.split('_')[0])
        teacher_inputs = dict(
            vid_feats=torch.randn(8, 512),  # placeholder if features pre‑extracted
            aud_feats=torch.randn(8, 128),
            interval_queries=torch.randn(4, 2)
        )

        return dict(I=video, A=wav, B=behavior, y=label, teacher_inputs=teacher_inputs)

class EasyComDataset(Dataset):
    """Active Speaker Localization dataset with world‑locked gaze vectors."""
    def __init__(self, root, csv_list):
        self.root = root
        self.samples = [l.strip() for l in open(csv_list)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_id = self.samples[idx]
        video = torch.load(os.path.join(self.root, 'frames', f'{vid_id}.pt'))
        audio = torch.load(os.path.join(self.root, 'audio', f'{vid_id}.pt'))
        gaze = torch.load(os.path.join(self.root, 'gaze', f'{vid_id}.pt'))  # [T,3]
        label = torch.load(os.path.join(self.root, 'labels', f'{vid_id}.pt'))
        return dict(I_seq=video, A_seq=audio, B_seq=gaze, y_asl=label)

class AEADataset(Dataset):
    """Behavior Anticipation dataset with multimodal sensor data."""
    def __init__(self, root, csv_list):
        self.root = root
        self.samples = [l.strip() for l in open(csv_list)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_id = self.samples[idx]
        video = torch.load(os.path.join(self.root, 'video', f'{vid_id}.pt'))
        audio = torch.load(os.path.join(self.root, 'audio', f'{vid_id}.pt'))
        behavior = torch.load(os.path.join(self.root, 'behavior', f'{vid_id}.pt'))
        label = torch.load(os.path.join(self.root, 'labels', f'{vid_id}.pt'))
        return dict(I_seq=video, A_seq=audio, B_seq=behavior, y_ba=label)