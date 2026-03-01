import random
import numpy as np
import subprocess
import timm
import os
import cv2
import json
import base64
import io
import secrets
import qrcode

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

# WebAuthn
from fido2.server import Fido2Server
from fido2.webauthn import (
    PublicKeyCredentialRpEntity,
    PublicKeyCredentialDescriptor,
    AttestedCredentialData
)

# ---------- BASE64 HELPERS ----------

def b64encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

def b64decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# ---------- CHALLENGE SENTENCE ----------

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

# ---------- WebAuthn Server ----------

rp = PublicKeyCredentialRpEntity(
    id="192.168.0.4",
    name="Interview Verification"
)

webauthn_server = Fido2Server(rp)

# ---------- QR SESSION STORE ----------

qr_sessions = {}


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        challenge=generate_challenge_sentence()
    )


# =========================================================
# DEVICE REGISTRATION
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

    credential["id"] = b64decode(credential["id"])
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
# LOCAL DEVICE AUTHENTICATION
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


@app.route("/auth_complete", methods=["POST"])
def auth_complete():

    credential = request.json["credential"]
    state = session["state"]
    username = session["username"]

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

    cred_path = os.path.join(UPLOAD_FOLDER, username, "webauthn.json")

    with open(cred_path) as f:
        stored = json.load(f)

    stored_credential = AttestedCredentialData(
        b64decode(stored["credential_data"])
    )

    webauthn_server.authenticate_complete(
        state,
        [stored_credential],
        credential
    )

    session["device_authenticated"] = True

    return "Device authenticated ✅"


# =========================================================
# QR SESSION CREATION (LAPTOP)
# =========================================================

@app.route("/qr_create", methods=["POST"])
def qr_create():

    username = request.json["username"]

    sid = secrets.token_urlsafe(16)

    qr_sessions[sid] = {
        "username": username,
        "authenticated": False
    }

    mobile_url = f"http://192.168.0.4:5000/qr_mobile/{sid}"

    img = qrcode.make(mobile_url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    qr_base64 = base64.b64encode(buf.read()).decode()

    return jsonify({
        "session_id": sid,
        "qr": f"data:image/png;base64,{qr_base64}"
    })



# =========================================================
# MOBILE PAGE (PHONE)
# =========================================================

@app.route("/qr_mobile/<sid>")
def qr_mobile_page(sid):

    return f"""
    <h2>Verify with Biometrics</h2>
    <button onclick="startAuth()">Authenticate</button>

    <script>
    function b64ToBuf(b64) {{
      b64 = b64.replace(/-/g, "+").replace(/_/g, "/");
      return Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    }}

    function bufToB64(buf) {{
      return btoa(String.fromCharCode(...new Uint8Array(buf)))
        .replace(/\\+/g, "-").replace(/\\//g, "_").replace(/=/g, "");
    }}

    async function startAuth() {{

        const r = await fetch("/qr_auth_begin/{sid}", {{method: "POST"}});
        const options = await r.json();

        options.challenge = b64ToBuf(options.challenge);
        options.allowCredentials.forEach(c => c.id = b64ToBuf(c.id));

        const cred = await navigator.credentials.get({{ publicKey: options }});

        const rawId = bufToB64(cred.rawId);

        const data = {{
            id: rawId,
            rawId: rawId,
            type: cred.type,
            response: {{
                authenticatorData: bufToB64(cred.response.authenticatorData),
                clientDataJSON: bufToB64(cred.response.clientDataJSON),
                signature: bufToB64(cred.response.signature),
                userHandle: null
            }}
        }};

        await fetch("/qr_auth_complete/{sid}", {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: JSON.stringify({{credential: data}})
        }});

        alert("Biometric verified. Return to laptop.");
    }}
    </script>
    """
    
@app.route("/qr_auth_begin/<sid>", methods=["POST"])
def qr_auth_begin(sid):

    data = qr_sessions.get(sid)
    if not data:
        return "Invalid session", 400

    username = data["username"]
    cred_path = os.path.join(UPLOAD_FOLDER, username, "webauthn.json")

    if not os.path.exists(cred_path):
        return "Device not registered", 400

    with open(cred_path) as f:
        cred_data = json.load(f)

    credential = PublicKeyCredentialDescriptor(
        id=b64decode(cred_data["credential_id"]),
        type="public-key"
    )

    options, state = webauthn_server.authenticate_begin([credential])

    session["state"] = state
    session["username"] = username
    session["qr_sid"] = sid

    def encode(obj):
        if isinstance(obj, bytes):
            return b64encode(obj)
        if isinstance(obj, dict):
            return {k: encode(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [encode(v) for v in obj]
        return obj

    return jsonify(encode(dict(options.public_key)))


@app.route("/qr_auth_complete/<sid>", methods=["POST"])
def qr_auth_complete(sid):

    credential = request.json["credential"]
    state = session["state"]
    username = session["username"]

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

    cred_path = os.path.join(UPLOAD_FOLDER, username, "webauthn.json")

    with open(cred_path) as f:
        stored = json.load(f)

    stored_credential = AttestedCredentialData(
        b64decode(stored["credential_data"])
    )

    webauthn_server.authenticate_complete(
        state,
        [stored_credential],
        credential
    )

    qr_sessions[sid]["authenticated"] = True

    return "OK"
    
    
@app.route("/qr_status/<sid>")
def qr_status(sid):

    data = qr_sessions.get(sid)

    if data and data["authenticated"]:
        session["device_authenticated"] = True

    return jsonify({"authenticated": data and data["authenticated"]})


# =========================================================
# VIDEO PROCESSING
# =========================================================

@app.route("/upload", methods=["POST"])
def upload_video():

    video = request.files["video"]
    username = request.form["username"]
    mode = request.form["mode"]

    if mode == "verify" and not session.get("device_authenticated"):
        return "Authenticate device first ❌"

    user_folder = os.path.join(UPLOAD_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)

    save_path = os.path.join(user_folder, "recorded_video.webm")
    video.save(save_path)

    frames = extract_frames(save_path)

    face_embedding = None

    for frame in frames:
        face = detect_face(frame)
        if face is not None:
            face_path = os.path.join(user_folder, "detected_face.jpg")
            cv2.imwrite(face_path, face)
            face_embedding = get_face_embedding(face_path)
            break

    if face_embedding is None:
        return "No face detected ❌"

    audio_path = os.path.join(user_folder, "temp_audio.wav")

    subprocess.run(
        ["ffmpeg", "-i", save_path, "-ar", "16000", "-ac", "1", audio_path, "-y"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    voice_embedding = get_voice_embedding(audio_path)

    face_ref_path = os.path.join(user_folder, "face_reference.npy")
    voice_ref_path = os.path.join(user_folder, "voice_reference.npy")

    if mode == "enroll":
        np.save(face_ref_path, face_embedding)
        np.save(voice_ref_path, voice_embedding)
        return "Enrollment Successful ✅"

    stored_face = np.load(face_ref_path)
    stored_voice = np.load(voice_ref_path)

    face_similarity = cosine_similarity(stored_face, face_embedding)
    voice_similarity = cosine_similarity(stored_voice, voice_embedding)

    video_tensor = process_video(save_path)
    video_prob = check_video_authenticity(video_tensor)

    audio_tensor = process_audio(audio_path)
    audio_prob = check_audio_authenticity(audio_tensor)

    if (
        face_similarity > 0.65 and
        voice_similarity > 0.75 and
        video_prob > 0.40 and
        audio_prob > 0.40
    ):
        return "FINAL VERIFIED & AUTHENTIC ✅"
    else:
        return "FINAL REJECTED ❌"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        ssl_context="adhoc"
    )