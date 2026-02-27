import cv2
import torch
import numpy as np
from torchvision import transforms


# =========================================================
# 1️⃣ FRAME EXTRACTION (For Face Verification)
# =========================================================

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return []

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_count += 1

    cap.release()
    print(f"Total frames extracted: {frame_count}")
    return frames


# =========================================================
# 2️⃣ FACE DETECTION (For Face Similarity)
# =========================================================

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    face = frame[y:y+h, x:x+w]

    return face


# =========================================================
# 3️⃣ VIDEO PROCESSING (For Authenticity Model)
# =========================================================

NUM_FRAMES = 5
IMG_SIZE = 224

video_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 1), NUM_FRAMES).astype(int)

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        if not ret:
            frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, _ = rgb.shape
        m = min(h, w)
        crop = rgb[h//2-m//2:h//2+m//2,
                   w//2-m//2:w//2+m//2]

        frames.append(video_transform(crop))

    cap.release()

    return torch.stack(frames).unsqueeze(0)