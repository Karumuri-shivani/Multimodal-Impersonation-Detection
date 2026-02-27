import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MTCNN (better detector than Haar)
mtcnn = MTCNN(image_size=160, margin=0, device=device)

# Initialize Face Embedding Model
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(image_path):
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect and align face
    face = mtcnn(img)

    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    # Generate embedding
    with torch.no_grad():
        embedding = model(face)

    return embedding.cpu().numpy()
