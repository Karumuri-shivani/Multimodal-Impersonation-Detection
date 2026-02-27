import torch
import torchaudio
from speechbrain.inference import EncoderClassifier

# Load pretrained ECAPA model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa",
    run_opts={"device": "cpu"}
)

def get_voice_embedding(audio_path):
    signal, fs = torchaudio.load(audio_path)

    # Convert to mono if needed
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    embedding = classifier.encode_batch(signal)

    return embedding.squeeze().detach().numpy()
