import random
import numpy as np
import subprocess
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

# ================= BASE64 HELPERS =================

def b64encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

def b64decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


# ================= CHALLENGE SENTENCE =================

def generate_challenge_sentence():
    sentences = [
        "Today I confirm that I am the authorized candidate.",
        "I am verifying my identity for this session.",
        "This verification ensures I am the genuine applicant."
    ]
    return random.choice(sentences)


app = Flask(__name__, template_folder="../frontend")
app.secret_key = "dev-secret"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= WebAuthn Server =================

rp = PublicKeyCredentialRpEntity(
    id="eugena-uncitizenlike-linwood.ngrok-free.dev",  # CHANGE if your IP changes
    name="Interview Verification"
)

webauthn_server = Fido2Server(rp)

# ================= QR SESSION STORE =================

qr_sessions = {}


# ======================================================
# HOME
# ======================================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        challenge=generate_challenge_sentence()
    )


# ======================================================
# QR — CREATE REGISTRATION SESSION
# ======================================================

@app.route("/qr_register_create", methods=["POST"])
def qr_register_create():

    username = request.json["username"]
    sid = secrets.token_urlsafe(16)

    qr_sessions[sid] = {
        "username": username,
        "type": "register",
        "authenticated": False
    }

    host = request.host_url.rstrip("/")
    mobile_url = f"{host}/qr_mobile/{sid}"

    img = qrcode.make(mobile_url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "session_id": sid,
        "qr": f"data:image/png;base64,{qr}"
    })


# ======================================================
# QR — CREATE AUTH SESSION
# ======================================================

@app.route("/qr_auth_create", methods=["POST"])
def qr_auth_create():

    username = request.json["username"]
    sid = secrets.token_urlsafe(16)

    qr_sessions[sid] = {
        "username": username,
        "type": "auth",
        "authenticated": False
    }

    host = request.host_url.rstrip("/")
    mobile_url = f"{host}/qr_mobile/{sid}"

    img = qrcode.make(mobile_url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "session_id": sid,
        "qr": f"data:image/png;base64,{qr}"
    })


# ======================================================
# PHONE PAGE
# ======================================================

@app.route("/qr_mobile/<sid>")
def qr_mobile(sid):

    return f"""
<h2>Phone Biometric Verification</h2>
<button onclick="go()">Continue</button>

<script>

function b64ToBuf(b64){{
 b64=b64.replace(/-/g,'+').replace(/_/g,'/');
 return Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
}}

function bufToB64(buf){{
 return btoa(String.fromCharCode(...new Uint8Array(buf)))
 .replace(/\\+/g,'-').replace(/\\//g,'_').replace(/=/g,'');
}}

async function go(){{
 const r=await fetch('/qr_webauthn_begin/{sid}',{{method:'POST'}});
 const opt=await r.json();

 opt.challenge=b64ToBuf(opt.challenge);
 if(opt.user) opt.user.id=b64ToBuf(opt.user.id);
 if(opt.allowCredentials)
   opt.allowCredentials.forEach(c=>c.id=b64ToBuf(c.id));

 let cred;

 if(opt.rp){{
   cred=await navigator.credentials.create({{publicKey:opt}});
 }} else {{
   cred=await navigator.credentials.get({{publicKey:opt}});
 }}

 const raw=bufToB64(cred.rawId);

 const data={{
  id:raw,
  rawId:raw,
  type:cred.type,
  response:{{
    clientDataJSON:bufToB64(cred.response.clientDataJSON),
    attestationObject:cred.response.attestationObject
      ? bufToB64(cred.response.attestationObject)
      : null,
    authenticatorData:cred.response.authenticatorData
      ? bufToB64(cred.response.authenticatorData)
      : null,
    signature:cred.response.signature
      ? bufToB64(cred.response.signature)
      : null
  }}
 }};

 await fetch('/qr_webauthn_complete/{sid}',{{
  method:'POST',
  headers:{{'Content-Type':'application/json'}},
  body:JSON.stringify({{credential:data}})
 }});

 alert("Success. Return to laptop.");
}}
</script>
"""


# ======================================================
# BEGIN WEBAUTHN
# ======================================================

@app.route("/qr_webauthn_begin/<sid>", methods=["POST"])
def qr_webauthn_begin(sid):

    data = qr_sessions[sid]
    username = data["username"]

    if data["type"] == "register":

        user = {
            "id": username.encode(),
            "name": username,
            "displayName": username
        }

        options, state = webauthn_server.register_begin(user)

    else:
        cred_path = os.path.join(UPLOAD_FOLDER, username, "webauthn.json")

        with open(cred_path) as f:
            c = json.load(f)

        credential = PublicKeyCredentialDescriptor(
            id=b64decode(c["credential_id"]),
            type="public-key"
        )

        options, state = webauthn_server.authenticate_begin([credential])

    session["state"] = state
    session["username"] = username
    session["sid"] = sid

    def enc(o):
        if isinstance(o, bytes): return b64encode(o)
        if isinstance(o, dict): return {k: enc(v) for k, v in o.items()}
        if isinstance(o, list): return [enc(v) for v in o]
        return o

    return jsonify(enc(dict(options.public_key)))


# ======================================================
# COMPLETE WEBAUTHN
# ======================================================

@app.route("/qr_webauthn_complete/<sid>", methods=["POST"])
def qr_webauthn_complete(sid):

    cred = request.json["credential"]
    username = session["username"]
    state = session["state"]

    cred["id"] = b64decode(cred["id"])
    cred["rawId"] = b64decode(cred["rawId"])

    for k, v in cred["response"].items():
        if v:
            cred["response"][k] = b64decode(v)

    if qr_sessions[sid]["type"] == "register":

        auth = webauthn_server.register_complete(state, cred)

        user_folder = os.path.join(UPLOAD_FOLDER, username)
        os.makedirs(user_folder, exist_ok=True)

        with open(os.path.join(user_folder, "webauthn.json"), "w") as f:
            json.dump({
                "credential_id": b64encode(auth.credential_data.credential_id),
                "credential_data": b64encode(bytes(auth.credential_data))
            }, f)

    else:

        cred_path = os.path.join(UPLOAD_FOLDER, username, "webauthn.json")

        with open(cred_path) as f:
            stored = json.load(f)

        stored_cred = AttestedCredentialData(
            b64decode(stored["credential_data"])
        )

        webauthn_server.authenticate_complete(
            state, [stored_cred], cred
        )

    qr_sessions[sid]["authenticated"] = True

    return "OK"


# ======================================================
# QR STATUS POLLING
# ======================================================

@app.route("/qr_status/<sid>")
def qr_status(sid):

    data = qr_sessions.get(sid)

    if data and data["authenticated"]:
        session["device_authenticated"] = True

    return jsonify({"authenticated": data and data["authenticated"]})


# ======================================================
# VIDEO PROCESSING (UNCHANGED)
# ======================================================

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

    subprocess.run([
        "ffmpeg",
        "-i", save_path,
        "-vn",                 # ❗ ignore video stream
        "-ac", "1",            # mono
        "-ar", "16000",        # 16 kHz
        "-acodec", "pcm_s16le",# clean WAV PCM
        audio_path,
        "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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

    face_similarity = float(face_similarity)
    voice_similarity = float(voice_similarity)
    video_prob = float(video_prob)
    audio_prob = float(audio_prob)

    # ------------------------------
    # Adaptive threshold for recorded videos
    # ------------------------------

    if mode == "verify_recorded":
        voice_threshold = 0.40
    else:
        voice_threshold = 0.75

    result = (
        face_similarity > 0.65 and
        voice_similarity > voice_threshold and
        video_prob > 0.40 and
        audio_prob > 0.40
    )

    message = f"""
    Face similarity: {face_similarity:.3f}
    Voice similarity: {voice_similarity:.3f}
    Video authenticity: {video_prob:.3f}
    Audio authenticity: {audio_prob:.3f}

    Final result: {"VERIFIED ✅" if result else "REJECTED ❌"}
    """

    return message


# ======================================================
# RUN SERVER
# ======================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        ssl_context="adhoc"
    )