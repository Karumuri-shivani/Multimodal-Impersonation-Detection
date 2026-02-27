import torch
import torch.nn as nn
import timm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

VIDEO_MODEL_PATH = os.path.join(MODEL_DIR, "best_video_model.pth")
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "best_audio_model.pth")


# ================= VIDEO MODEL =================

class VideoViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,   # MUST be False (same as Stage 4)
            num_classes=0
        )
        self.head = nn.Linear(768, 1)

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)
        feat = self.vit(x)
        feat = feat.view(B, F, -1).mean(dim=1)
        return self.head(feat)


# ================= AUDIO MODEL =================

class AudioViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,   # MUST be False (same as Stage 4)
            num_classes=0,
            in_chans=1
        )
        self.head = nn.Linear(768, 1)

    def forward(self, x):
        feat = self.vit(x)
        return self.head(feat)


# ================= LOAD MODELS =================

video_model = VideoViT().to(device)
video_model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=device))
video_model.eval()

audio_model = AudioViT().to(device)
audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
audio_model.eval()

print("✅ Authenticity models loaded successfully")


# ================= INFERENCE =================

def check_video_authenticity(video_tensor):
    with torch.no_grad():
        video_tensor = video_tensor.to(device)
        output = video_model(video_tensor)
        prob = torch.sigmoid(output).item()
    return prob  # 1 = REAL, 0 = FAKE


def check_audio_authenticity(audio_tensor):
    with torch.no_grad():
        audio_tensor = audio_tensor.to(device)
        output = audio_model(audio_tensor)
        prob = torch.sigmoid(output).item()
    return prob  # 1 = REAL, 0 = FAKE
