import numpy as np
from voice_embedding import get_voice_embedding

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# First recording → reference
ref_embedding = get_voice_embedding("uploads/audio.wav")
np.save("uploads/voice_reference.npy", ref_embedding)
print("Reference voice saved.")

# Simulate second recording (use same file for now)
new_embedding = get_voice_embedding("uploads/audio.wav")

similarity = cosine_similarity(ref_embedding, new_embedding)
print("Voice similarity:", similarity)

if similarity > 0.75:
    print("VOICE VERIFIED ✅")
else:
    print("VOICE REJECTED ❌")
