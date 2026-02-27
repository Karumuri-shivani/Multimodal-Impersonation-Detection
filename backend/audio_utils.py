import torch
import numpy as np
import librosa
import torch.nn.functional as F

# ---------------- AUTHENTICITY AUDIO PROCESSING ----------------

SAMPLE_RATE = 16000
AUDIO_SECONDS = 5
N_MELS = 128
IMG_SIZE = 224

def process_audio(video_path):
    try:
        y, _ = librosa.load(video_path, sr=SAMPLE_RATE, mono=True)
    except:
        y = np.zeros(SAMPLE_RATE * AUDIO_SECONDS)

    max_len = SAMPLE_RATE * AUDIO_SECONDS
    y = y[:max_len]

    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

    mel = F.interpolate(
        mel.unsqueeze(0),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    return mel.unsqueeze(0)  # (1, 1, 224, 224)
