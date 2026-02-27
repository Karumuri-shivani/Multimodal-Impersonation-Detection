# import os
# import json
# import base64

# from flask import Flask, render_template, request, jsonify, session
# from fido2.server import Fido2Server
# from fido2.webauthn import PublicKeyCredentialRpEntity, PublicKeyCredentialDescriptor

# # ---------------------------
# # Base64 helpers
# # ---------------------------

# def b64encode(b: bytes) -> str:
#     return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

# def b64decode(s: str) -> bytes:
#     return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# # ---------------------------
# # App setup
# # ---------------------------

# app = Flask(__name__, template_folder="../frontend")
# app.secret_key = "dev-secret"

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# rp = PublicKeyCredentialRpEntity(id="localhost", name="Test App")
# server = Fido2Server(rp)


# # ---------------------------
# # Home
# # ---------------------------

# @app.route("/")
# def index():
#     return render_template("index.html")


# # ---------------------------
# # REGISTER BEGIN
# # ---------------------------

# @app.route("/register_begin", methods=["POST"])
# def register_begin():

#     username = request.json["username"]

#     user = {
#         "id": username.encode(),
#         "name": username,
#         "displayName": username,
#     }

#     options, state = server.register_begin(user)

#     session["state"] = state
#     session["username"] = username

#     # Convert bytes → base64
#     def encode(obj):
#         if isinstance(obj, bytes):
#             return b64encode(obj)
#         if isinstance(obj, dict):
#             return {k: encode(v) for k, v in obj.items()}
#         if isinstance(obj, list):
#             return [encode(v) for v in obj]
#         return obj

#     return jsonify(encode(dict(options.public_key)))


# # ---------------------------
# # REGISTER COMPLETE
# # ---------------------------

# @app.route("/register_complete", methods=["POST"])
# def register_complete():

#     data = request.json
#     username = session.get("username")
#     state = session.get("state")

#     if not username or not data:
#         return "Session expired ❌", 400

#     try:
#         # Decode ONLY binary fields
#         data["rawId"] = b64decode(data["rawId"])
#         data["response"]["attestationObject"] = b64decode(
#             data["response"]["attestationObject"]
#         )
#         data["response"]["clientDataJSON"] = b64decode(
#             data["response"]["clientDataJSON"]
#         )

#         # ⭐ DO NOT TOUCH data["id"]

#         auth_data = server.register_complete(state, data)

#     except Exception as e:
#         return f"Registration failed ❌ {e}", 400

#     cred_data = auth_data.credential_data

#     user_folder = os.path.join(UPLOAD_FOLDER, username)
#     os.makedirs(user_folder, exist_ok=True)

#     with open(os.path.join(user_folder, "webauthn_credential.json"), "w") as f:
#         json.dump({
#             "credential_id": b64encode(cred_data.credential_id),
#             "public_key": b64encode(cred_data.public_key),
#             "sign_count": auth_data.sign_count
#         }, f)

#     return "Device Registered ✅"


# # ---------------------------
# # AUTH BEGIN
# # ---------------------------

# @app.route("/auth_begin", methods=["POST"])
# def auth_begin():

#     username = request.json["username"]

#     cred_path = os.path.join(UPLOAD_FOLDER, username, "cred.json")

#     if not os.path.exists(cred_path):
#         return "Not registered", 400

#     with open(cred_path) as f:
#         cred_data = json.load(f)

#     credential = PublicKeyCredentialDescriptor(
#         id=b64decode(cred_data["id"])
#     )

#     options, state = server.authenticate_begin([credential])

#     session["state"] = state

#     def encode(obj):
#         if isinstance(obj, bytes):
#             return b64encode(obj)
#         if isinstance(obj, dict):
#             return {k: encode(v) for k, v in obj.items()}
#         if isinstance(obj, list):
#             return [encode(v) for v in obj]
#         return obj

#     return jsonify(encode(dict(options.public_key)))


# # ---------------------------
# # AUTH COMPLETE
# # ---------------------------

# @app.route("/auth_complete", methods=["POST"])
# def auth_complete():

#     data = request.json["credential"]
#     state = session["state"]

#     cred_id = b64decode(data["rawId"])

#     data["rawId"] = cred_id
#     data["response"]["authenticatorData"] = b64decode(
#         data["response"]["authenticatorData"]
#     )
#     data["response"]["clientDataJSON"] = b64decode(
#         data["response"]["clientDataJSON"]
#     )
#     data["response"]["signature"] = b64decode(
#         data["response"]["signature"]
#     )

#     server.authenticate_complete(state, [], data)

#     return "Authenticated ✅"

# if __name__ == "__main__":
#     app.run(debug=True)

import random
import numpy as np
import subprocess
import torch
import os
import cv2
import json
import base64

from flask import Flask, render_template, request, jsonify, session

from authenticity_models import (
    check_video_authenticity,
    check_audio_authenticity
)
from voice_embedding import get_voice_embedding
from similarity import cosine_similarity
from face_embedding import get_face_embedding
from video_utils import extract_frames, detect_face, process_video
from audio_utils import process_audio

# 🔐 WebAuthn
from fido2.server import Fido2Server
from fido2.webauthn import PublicKeyCredentialRpEntity, PublicKeyCredentialDescriptor
from fido2.webauthn import AttestedCredentialData

# ---------------- BASE64 HELPERS ----------------

def b64encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

def b64decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# ---------------- CHALLENGE SENTENCE ----------------

def generate_challenge_sentence():
    sentences = [
        "Today I confirm that I am the authorized candidate for this interview process.",
        "I am verifying my identity for this session.",
        "This verification step ensures that I am the genuine applicant.",
        "I understand that this system uses face and voice verification."
    ]
    return random.choice(sentences)


app = Flask(__name__, template_folder="../frontend")
app.secret_key = "dev-secret"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔐 WebAuthn Server (version)
rp = PublicKeyCredentialRpEntity(
    id="localhost",
    name="Interview Verification"
)
webauthn_server = Fido2Server(rp)


# ---------------- HOME ----------------

@app.route("/")
def index():
    challenge_sentence = generate_challenge_sentence()
    return render_template("index.html", challenge=challenge_sentence)


# =========================================================
# 🔐 REGISTER DEVICE
# =========================================================

@app.route("/register_begin", methods=["POST"])
def register_begin():

    username = request.json["username"]

    user = {
        "id": username.encode(),
        "name": username,
        "displayName": username,
    }

    options, state = webauthn_server.register_begin(
    user,
    authenticator_attachment="platform",
    user_verification="preferred",
)

    session["state"] = state
    session["username"] = username

    def encode(obj):
        if isinstance(obj, bytes):
            return b64encode(obj)
        if isinstance(obj, dict):
            return {k: encode(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [encode(v) for v in obj]
        return obj

    return jsonify(encode(dict(options.public_key)))


@app.route("/register_complete", methods=["POST"])
def register_complete():

    credential = request.json["credential"]
    state = session["state"]
    username = session["username"]

    credential["id"] = b64decode(credential["id"])          # ⭐ ADD THIS
    credential["rawId"] = b64decode(credential["rawId"])

    credential["response"]["attestationObject"] = b64decode(
        credential["response"]["attestationObject"]
    )

    credential["response"]["clientDataJSON"] = b64decode(
        credential["response"]["clientDataJSON"]
)

    auth_data = webauthn_server.register_complete(state, credential)

    cred_data = auth_data.credential_data

    user_folder = os.path.join(UPLOAD_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)
    with open(os.path.join(user_folder, "webauthn.json"), "w") as f:
        json.dump({
            "credential_id": b64encode(cred_data.credential_id),
            "credential_data": b64encode(bytes(cred_data)),
            "sign_count": auth_data.counter
        }, f)

    return "Device registered ✅"


# =========================================================
# 🔐 AUTHENTICATE DEVICE
# =========================================================

@app.route("/auth_begin", methods=["POST"])
def auth_begin():

    username = request.json["username"]
    cred_path = os.path.join(UPLOAD_FOLDER, username, "webauthn.json")

    if not os.path.exists(cred_path):
        return "Device not registered ❌", 400

    with open(cred_path) as f:
        cred_data = json.load(f)

    credential = PublicKeyCredentialDescriptor(
    id=b64decode(cred_data["credential_id"]),
    type="public-key"
)

    options, state = webauthn_server.authenticate_begin([credential])

    session["state"] = state

    def encode(obj):
        if isinstance(obj, bytes):
            return b64encode(obj)
        if isinstance(obj, dict):
            return {k: encode(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [encode(v) for v in obj]
        return obj

    return jsonify(encode(dict(options.public_key)))


@app.route("/auth_complete", methods=["POST"])
def auth_complete():

    credential = request.json["credential"]
    state = session["state"]
    username = session["username"]

    # 🔐 Decode fields
    credential["id"] = b64decode(credential["id"])
    credential["rawId"] = b64decode(credential["rawId"])

    credential["response"]["authenticatorData"] = b64decode(
        credential["response"]["authenticatorData"]
    )
    credential["response"]["clientDataJSON"] = b64decode(
        credential["response"]["clientDataJSON"]
    )
    credential["response"]["signature"] = b64decode(
        credential["response"]["signature"]
    )

    # 🔐 Load stored credential
    cred_path = os.path.join(UPLOAD_FOLDER, username, "webauthn.json")

    if not os.path.exists(cred_path):
        return "Device not registered ❌", 400

    with open(cred_path) as f:
        stored = json.load(f)

    stored_credential = AttestedCredentialData(
        b64decode(stored["credential_data"])
    )

    # 🔐 VERIFY against stored credential
    webauthn_server.authenticate_complete(
        state,
        [stored_credential],   # ⭐ CORRECT OBJECT TYPE
        credential
    )

    session["device_authenticated"] = True

    return "Device authenticated ✅"


# =========================================================
# 🎥 YOUR EXISTING UPLOAD ROUTE
# =========================================================

@app.route("/upload", methods=["POST"])
def upload_video():

    video = request.files["video"]
    username = request.form["username"]
    mode = request.form["mode"]

    # 🔐 Only require device authentication for verification
    if mode == "verify" and not session.get("device_authenticated"):
        return "Authenticate device first ❌"

    video = request.files["video"]
    username = request.form["username"]
    mode = request.form["mode"]

    user_folder = os.path.join(UPLOAD_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)

    save_path = os.path.join(user_folder, "recorded_video.webm")
    video.save(save_path)

    # ---- YOUR ORIGINAL PROCESSING CONTINUES HERE ----
    return "Video received ✅"


if __name__ == "__main__":
    app.run(debug=True)