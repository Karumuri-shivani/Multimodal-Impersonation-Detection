# Multimodal Impersonation Detection System (MID-Net)

## Overview

MID-Net is a multimodal biometric authentication system designed to detect impersonation attempts using a combination of biometric similarity and media authenticity analysis. The system integrates face recognition, voice recognition, deepfake detection, and device-level authentication to provide a robust and secure verification pipeline.

---

## Key Features

* Multimodal authentication using face and voice biometrics
* Video deepfake detection using Vision Transformer (ViT)
* Audio deepfake detection using spectrogram-based Vision Transformer
* Face embeddings generated using FaceNet (InceptionResNetV1)
* Voice embeddings generated using ECAPA-TDNN
* QR-based device authentication using WebAuthn
* Decision-level fusion of multiple verification signals

---

## System Architecture

The system operates in three distinct phases:

### Enrollment Phase

* User provides a video sample containing both face and voice
* Face and voice embeddings are extracted
* Biometric templates are stored in the database
* No identity verification is performed during this phase

### Device Authentication (WebAuthn)

* QR code-based authentication between user device and system
* Public-private key cryptography ensures secure device verification
* Only registered devices are allowed to proceed

### Verification Phase

* User provides live video input
* Face and voice embeddings are extracted and compared with stored templates
* Video and audio authenticity are evaluated using trained models
* Final decision is made using threshold-based fusion of:

  * Face similarity
  * Voice similarity
  * Video authenticity score
  * Audio authenticity score
  * Device verification status

---

## Technologies Used

* Python
* PyTorch
* OpenCV
* Librosa
* Timm (Vision Transformer)
* MTCNN
* FaceNet (InceptionResNetV1)
* ECAPA-TDNN
* WebAuthn

---

## Model Details

* Video Model: Vision Transformer (ViT-B/16)
* Audio Model: Vision Transformer applied to Mel Spectrograms
* Loss Function: Binary Cross Entropy with Logits
* Optimizer: Adam
* Input Resolution: 224 × 224

---

## Dataset

* FakeAVCeleb Dataset

---

## Model Files

The trained `.pth` model files are not included in this repository due to size constraints.

To run the project:

1. Download the model files from: **[ADD DRIVE LINK HERE]**
2. Place them in the following directories:

   * `Video_Checkpoints/`
   * `Audio_Checkpoints/`

---

## Setup and Execution

```bash
pip install -r requirements.txt
python app.py
```

---

## Limitations

* Enrollment phase does not validate identity authenticity
* Audio deepfake detection performance is lower compared to video
* Authentication depends on access to the registered device
* High computational requirements limit real-time scalability

---

## Future Work

* Integration of liveness detection mechanisms
* Improved audio deepfake detection using advanced architectures
* Adaptive fusion strategies instead of fixed thresholds
* Scalable cloud-based deployment

---

## Conclusion

MID-Net demonstrates that combining multiple modalities—biometric similarity, media authenticity, and device verification—significantly enhances the robustness of authentication systems against impersonation and deepfake-based attacks.

---

## Contributors

* Shivani Karumuri
* Ruchitanjani Jella
* Prasanna Lakshmi Nakka
* Shravani Kondikopulla
* Charan Bolla

---

## Demo

[Deployment link will be added after deployment]
