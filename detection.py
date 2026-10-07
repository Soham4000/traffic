"""
Vehicle Detection, Tracking, Speed Estimation & Helmet Check — Streamlit App
-----------------------------------------------------------------------------
Detects cars, buses, trucks, motorcycles/scooters, and bicycles in an
uploaded video, tracks each one with a persistent ID (so it's counted once,
not once per frame), estimates rough speed, and — for two-wheelers — checks
whether the rider is wearing a helmet, highlighting violations in red.

NEW: Telegram alerts — violation snapshots can be sent automatically to a
Telegram chat/group (immediately, after human review, or both).

Run with:
    streamlit run vehicle_detection_app.py

-----------------------------------------------------------------------------
DEPENDENCIES
-----------------------------------------------------------------------------
    pip install streamlit opencv-python-headless torch torchvision ultralytics lapx
    pip install huggingface_hub requests

`lapx` is required by Ultralytics' ByteTrack tracker (used here for counting
and speed estimation) — without it, model.track() will raise an ImportError.

-----------------------------------------------------------------------------
TELEGRAM SETUP
-----------------------------------------------------------------------------
1. In Telegram, talk to @BotFather -> /newbot -> copy the bot token.
2. Send any message to your bot (or add it to a group and message there).
3. Open https://api.telegram.org/bot<TOKEN>/getUpdates and copy chat -> id
   (group IDs are negative numbers).
4. Provide the two values in ANY ONE of these ways:
   a) Type them into the app's sidebar ("Enter Telegram details manually"),
   b) a .env file next to this script (pip install python-dotenv):
          TELEGRAM_BOT_TOKEN=...
          TELEGRAM_CHAT_ID=...
   c) .streamlit/secrets.toml, or
   d) environment variables BEFORE running streamlit:

   Windows (PowerShell):
       $env:TELEGRAM_BOT_TOKEN="123456:ABC..."
       $env:TELEGRAM_CHAT_ID="987654321"

   Linux / macOS:
       export TELEGRAM_BOT_TOKEN="123456:ABC..."
       export TELEGRAM_CHAT_ID="987654321"

Never hardcode the token in this file.

-----------------------------------------------------------------------------
ABOUT HELMET DETECTION
-----------------------------------------------------------------------------
The default YOLO model (trained on COCO) only recognizes generic object
classes like "motorcycle" or "bicycle" — it has NO concept of "helmet".
This app therefore auto-downloads a SECOND, publicly available model
trained specifically for helmet/no-helmet detection (from Hugging Face:
Abs6187/Helmet-Detect-model), the first time it runs, and caches it locally
as `helmet_yolov8.pt` next to this script. No manual setup needed — just
an internet connection on first launch.

If you'd rather use your own helmet-detection model instead, just place a
file named exactly `helmet_yolov8.pt` in the same folder as this script
before running it — the app will use that instead of downloading one.

Dependency needed for the auto-download: `pip install huggingface_hub`
-----------------------------------------------------------------------------
"""

import os
import io
import sys
import time
import uuid
import json
import base64
import sqlite3
import zipfile
import tempfile
import datetime
import threading
from collections import deque, defaultdict

import streamlit as st

st.set_page_config(page_title="Vehicle Detection", layout="wide")
st.title("🚗 Vehicle Detection, Tracking & Helmet Check")
st.caption("Cars, buses, trucks, motorcycles/scooters, bicycles — with counting, speed, and helmet compliance")

PY_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

# ---------------------------------------------------------------------------
# Dependency check block
# ---------------------------------------------------------------------------
missing = []

try:
    import cv2
except ImportError as e:
    missing.append(("opencv-python-headless", str(e)))

try:
    import torch  # noqa: F401
except ImportError as e:
    missing.append(("torch", str(e)))

try:
    from ultralytics import YOLO
except ImportError as e:
    missing.append(("ultralytics", str(e)))

if missing:
    st.error(f"Running on Python {PY_VERSION} — some required packages failed to import.")
    for pkg, err in missing:
        st.write(f"**{pkg}** failed to import:")
        st.code(err)
    st.markdown("### How to fix")
    st.markdown(
        "1. `python -m pip install --upgrade pip`\n"
        "2. `python -m pip install --upgrade opencv-python-headless`\n"
        "3. `python -m pip install torch torchvision torchaudio` "
        "(or the nightly build if that fails on your Python version)\n"
        "4. `python -m pip install --upgrade ultralytics lapx`\n"
        "5. Restart: `streamlit run vehicle_detection_app.py`"
    )
    st.stop()

with st.expander("🔧 Environment info"):
    st.write(f"Python: {PY_VERSION}")
    st.write(f"OpenCV: {cv2.__version__}")
    st.write(f"PyTorch: {torch.__version__}")

# ---------------------------------------------------------------------------
# Optional dependencies. These are NOT required for core detection (vehicle/
# helmet detection stays 100% local YOLO models) — they only power auxiliary
# features: incident reports, log queries, notice drafting, second opinion on
# borderline cases, plate OCR, and Telegram alerts. Each feature checks
# availability and degrades gracefully if missing.
# ---------------------------------------------------------------------------
import numpy as np

ANTHROPIC_AVAILABLE = True
try:
    import anthropic
except ImportError:
    ANTHROPIC_AVAILABLE = False

EASYOCR_AVAILABLE = True
try:
    import easyocr
except ImportError:
    EASYOCR_AVAILABLE = False

REQUESTS_AVAILABLE = True
try:
    import requests
except ImportError:
    REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Vehicle classes (COCO ids used by the pretrained YOLO model)
# ---------------------------------------------------------------------------
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle/scooter", 5: "bus", 7: "truck"}
TWO_WHEELER_IDS = {1, 3}  # bicycle, motorcycle/scooter — the ones we check for helmets
PERSON_CLASS_ID = 0       # COCO "person" — used to locate the actual rider, far more
                          # reliable than guessing head position from the vehicle box alone

CLASS_COLORS = {
    1: (255, 200, 0),
    2: (0, 255, 0),
    3: (0, 165, 255),
    5: (255, 0, 255),
    7: (0, 0, 255),
}

RED = (0, 0, 255)
GREEN = (0, 200, 0)


@st.cache_resource
def load_vehicle_model(weights_name: str):
    return YOLO(weights_name)


@st.cache_resource
def load_helmet_model(path: str):
    return YOLO(path)


# ---------------------------------------------------------------------------
# Everything below runs automatically with fixed defaults — no settings panel.
# ---------------------------------------------------------------------------
MODEL_SIZE = "yolov8n.pt"
CONF_THRESHOLD = 0.35
ENABLE_SPEED = True
METERS_PER_100PX = 5.0  # rough calibration constant; adjust in code if your footage differs a lot

HELMET_WEIGHTS_FILENAME = "helmet_yolov8.pt"
# Class-name aliases used by the downloaded Abs6187 helmet model.
# Its data.yaml defines: accept-Helmet- and non-Helmet-.
HELMET_NO_HELMET_KEYWORDS = (
    "no helmet", "no_helmet", "no-helmet",
    "without helmet", "without_helmet", "without-helmet",
    "non helmet", "non_helmet", "non-helmet", "nonhelmet",
    "unhelmeted", "un-helmeted", "unhelmet",
)
HELMET_POSITIVE_KEYWORDS = (
    "helmet", "with helmet", "with_helmet", "helmeted",
    "accept helmet", "accept_helmet", "accept-helmet",
)
HELMET_INFERENCE_CONF = 0.35
NO_HELMET_CONF = 0.60
NO_HELMET_VOTE_WINDOW = 5
NO_HELMET_MIN_VOTES = 3

# Public pretrained helmet-detection model (used only if no local file already exists)
HF_HELMET_REPO_ID = "Abs6187/Helmet-Detect-model"
HF_HELMET_REPO_TYPE = "space"
HF_HELMET_FILENAME = "best.pt"

conf_threshold = CONF_THRESHOLD
enable_speed = ENABLE_SPEED
meters_per_100px = METERS_PER_100PX
helmet_no_helmet_keywords = HELMET_NO_HELMET_KEYWORDS
helmet_positive_keywords = HELMET_POSITIVE_KEYWORDS

# ---------------------------------------------------------------------------
# Persistent audit log (SQLite) — survives across app restarts, unlike
# st.session_state. This is the paper trail a real deployed system would
# need: every violation, when it happened, how confident the model was,
# which camera/location it came from, and what a human reviewer decided.
# ---------------------------------------------------------------------------
DB_PATH = "violations_log.db"

# Active-learning folders: every reviewed snapshot is saved here automatically,
# building a real, human-verified training dataset from actual usage — no
# separate manual labeling effort required.
CONFIRMED_VIOLATION_DIR = "training_data/confirmed_no_helmet"
CONFIRMED_FALSE_POSITIVE_DIR = "training_data/confirmed_false_positive"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS violations (
            violation_id TEXT PRIMARY KEY,
            timestamp TEXT,
            location TEXT,
            video_source TEXT,
            track_id INTEGER,
            vehicle_type TEXT,
            confidence REAL,
            review_status TEXT,
            plate_number TEXT,
            ai_second_opinion TEXT
        )
        """
    )
    # Migration for DBs created before these two columns existed.
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(violations)").fetchall()]
    if "plate_number" not in existing_cols:
        conn.execute("ALTER TABLE violations ADD COLUMN plate_number TEXT")
    if "ai_second_opinion" not in existing_cols:
        conn.execute("ALTER TABLE violations ADD COLUMN ai_second_opinion TEXT")
    conn.commit()
    conn.close()


def log_violation(violation_id, location, video_source, track_id, vehicle_type, confidence,
                   plate_number=None, ai_second_opinion=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO violations "
        "(violation_id, timestamp, location, video_source, track_id, vehicle_type, confidence, "
        "review_status, plate_number, ai_second_opinion) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            violation_id,
            datetime.datetime.now().isoformat(timespec="seconds"),
            location,
            video_source,
            track_id,
            vehicle_type,
            confidence,
            "pending",
            plate_number,
            ai_second_opinion,
        ),
    )
    conn.commit()
    conn.close()


def update_review_status(violation_id, status):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE violations SET review_status = ? WHERE violation_id = ?", (status, violation_id))
    conn.commit()
    conn.close()


def fetch_all_violations():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM violations ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_training_sample(jpeg_bytes, violation_id, accepted: bool):
    """Save a reviewed snapshot into the appropriate active-learning folder."""
    target_dir = CONFIRMED_VIOLATION_DIR if accepted else CONFIRMED_FALSE_POSITIVE_DIR
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, f"{violation_id}.jpg"), "wb") as f:
        f.write(jpeg_bytes)


init_db()

# ---------------------------------------------------------------------------
# Telegram alerts — send violation snapshots to a Telegram chat/group.
# Credentials come from environment variables (never hardcode them).
# Sending runs in a background thread so the video loop never stalls on
# network latency.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv   # optional: pip install python-dotenv
    load_dotenv()
except ImportError:
    pass


def _get_secret(name):
    """Look in env vars / .env first, then Streamlit secrets (.streamlit/secrets.toml)."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        val = st.secrets.get(name)
        if val:
            return str(val)
    except Exception:
        pass
    return None


TELEGRAM_BOT_TOKEN = _get_secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _get_secret("TELEGRAM_CHAT_ID")
TELEGRAM_AVAILABLE = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and REQUESTS_AVAILABLE)

TELEGRAM_MODE_OFF = "Off"
TELEGRAM_MODE_INSTANT = "Instantly when detected"
TELEGRAM_MODE_ACCEPTED = "Only after reviewer clicks Accept"
TELEGRAM_MODE_BOTH = "Both (instant + confirmation)"


def _send_telegram_photo(jpeg_bytes, caption):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        r = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000]},
            files={"photo": ("violation.jpg", jpeg_bytes, "image/jpeg")},
            timeout=20,
        )
        if not r.ok:
            print(f"Telegram error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"Telegram send failed: {e}")


def send_telegram_test():
    """Synchronous test so you can see the exact error, if any."""
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": "✅ Test from Vehicle Detection app"},
            timeout=15,
        )
        return r.ok, r.text
    except Exception as e:
        return False, str(e)


def send_telegram_alert(jpeg_bytes, location, vehicle_type, track_id, confidence,
                        plate_number=None, status="🟡 Pending review"):
    """Fire-and-forget Telegram photo alert. Safe to call even if Telegram
    isn't configured — it simply does nothing."""
    if not TELEGRAM_AVAILABLE:
        return
    caption = (
        "🚨 Helmet violation detected\n"
        f"📍 Location: {location}\n"
        f"🕒 Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"🏍️ Vehicle: {vehicle_type} (rider #{track_id})\n"
        f"📊 Confidence: {confidence * 100:.0f}%\n"
        f"🔢 Plate (OCR, unverified): {plate_number or 'not captured'}\n"
        f"📝 Status: {status}"
    )
    threading.Thread(
        target=_send_telegram_photo, args=(jpeg_bytes, caption), daemon=True
    ).start()


# ---------------------------------------------------------------------------
# Optional AI features (auxiliary only — none of these are part of the core
# vehicle/helmet detection pipeline, which stays 100% local YOLO).
# ---------------------------------------------------------------------------

ANTHROPIC_MODEL = "claude-sonnet-4-6"
# Confidence band around the NO_HELMET_CONF decision boundary considered
# "borderline" enough to warrant an optional second opinion.
BORDERLINE_LOW = NO_HELMET_CONF - 0.10
BORDERLINE_HIGH = NO_HELMET_CONF + 0.10


@st.cache_resource
def get_anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource
def get_ocr_reader():
    if not EASYOCR_AVAILABLE:
        return None
    try:
        return easyocr.Reader(["en"], gpu=False)
    except Exception:
        return None


def read_plate_text(jpeg_bytes):
    """Best-effort local OCR on a violation snapshot to extract a license
    plate string. Runs fully offline (no external API calls). Returns None
    if OCR isn't available or nothing plate-like is found."""
    reader = get_ocr_reader()
    if reader is None:
        return None
    try:
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        results = reader.readtext(img)
        # Heuristic: a plate string is usually short-ish, alphanumeric, and
        # the OCR was reasonably confident about it. Pick the best candidate.
        candidates = [
            text.strip() for (_, text, conf) in results
            if conf > 0.35 and 4 <= len(text.strip()) <= 12 and any(c.isalnum() for c in text)
        ]
        if candidates:
            return max(candidates, key=len)
    except Exception:
        pass
    return None


def ai_second_opinion(jpeg_bytes):
    """For borderline-confidence cases only: ask a vision-capable LLM for a
    second read on the same snapshot. This is purely an EXTRA data point
    shown to the human reviewer — it never decides the violation itself,
    and is not called for confidently-decided cases."""
    client = get_anthropic_client()
    if client is None:
        return None
    try:
        img_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": (
                        "Is the two-wheeler rider in this image wearing a helmet? "
                        "Answer with exactly one word: HELMET, NOHELMET, or UNCLEAR."
                    )},
                ],
            }],
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        return text.strip().upper()
    except Exception:
        return None


def generate_incident_report(rows):
    """Summarize the persistent violation log into a short plain-English
    report using an LLM. Purely a reporting layer — reads already-logged
    data, doesn't affect detection."""
    client = get_anthropic_client()
    if client is None or not rows:
        return None
    data_str = "\n".join(
        f"{r['timestamp']} | {r['location']} | {r['vehicle_type']} | "
        f"confidence={r['confidence']:.2f} | status={r['review_status']}"
        for r in rows[:200]
    )
    prompt = (
        "Here is a traffic helmet-violation log (timestamp | location | vehicle type | "
        "confidence | review status):\n\n" + data_str +
        "\n\nWrite a short (3-5 sentence) plain-English summary highlighting trends, "
        "peak times, and notable patterns. Only state facts supported by this data — "
        "do not invent numbers not present here."
    )
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    except Exception as e:
        return f"(Report generation failed: {e})"


def answer_log_question(question, rows):
    """Answer a free-text question about the violation log using an LLM,
    grounded strictly in the logged data provided as context."""
    client = get_anthropic_client()
    if client is None:
        return None
    data_str = "\n".join(
        f"{r['timestamp']} | {r['location']} | {r['vehicle_type']} | "
        f"confidence={r['confidence']:.2f} | status={r['review_status']}"
        for r in rows[:500]
    )
    prompt = (
        "Violation log data (timestamp | location | vehicle type | confidence | review status):\n\n"
        + data_str + f"\n\nQuestion: {question}\n\n"
        "Answer using ONLY the data above. If the data doesn't contain enough information "
        "to answer, say so plainly instead of guessing."
    )
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    except Exception as e:
        return f"(Query failed: {e})"


def draft_violation_notice(row):
    """Draft formal notice text for an ACCEPTED (human-confirmed) violation.
    The AI only writes up the record — it never decides the violation, and
    nothing gets sent anywhere automatically; this is text for a human to
    review before use."""
    client = get_anthropic_client()
    if client is None:
        return None
    prompt = (
        "Draft a short, formal traffic violation notice based on this confirmed record:\n"
        f"Date/Time: {row['timestamp']}\n"
        f"Location: {row['location']}\n"
        f"Vehicle type: {row['vehicle_type']}\n"
        f"Violation: Riding without a helmet\n"
        f"Model confidence: {row['confidence']*100:.0f}%\n"
        f"Plate (if read): {row.get('plate_number') or 'not captured'}\n\n"
        "Keep it under 120 words, formal tone, and include a line noting this was "
        "reviewed and confirmed by a human operator before issuance."
    )
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    except Exception as e:
        return f"(Notice generation failed: {e})"


def ensure_helmet_weights() -> str | None:
    """
    Returns a local path to helmet-detection weights, downloading a public
    pretrained model from Hugging Face on first run if no local file exists
    yet. Returns None if neither a local file nor a successful download
    is available.
    """
    if os.path.exists(HELMET_WEIGHTS_FILENAME):
        return HELMET_WEIGHTS_FILENAME
    try:
        from huggingface_hub import hf_hub_download
        with st.spinner("Downloading a pretrained helmet-detection model (first run only)..."):
            downloaded_path = hf_hub_download(
                repo_id=HF_HELMET_REPO_ID,
                repo_type=HF_HELMET_REPO_TYPE,
                filename=HF_HELMET_FILENAME,
            )
        return downloaded_path
    except Exception as e:
        st.caption(f"⚠️ Could not auto-download a helmet model: {e}")
        return None


helmet_model = None
helmet_weights_path = ensure_helmet_weights()
if helmet_weights_path:
    try:
        helmet_model = load_helmet_model(helmet_weights_path)
        st.caption(f"✅ Helmet model loaded — classes: {helmet_model.names}")
    except Exception as e:
        st.caption(f"⚠️ Found a helmet model file but couldn't load it: {e}")
else:
    st.caption(
        "ℹ️ Helmet check inactive — auto-download failed and no local "
        f"`{HELMET_WEIGHTS_FILENAME}` file was found. Vehicle detection, "
        "tracking, and speed still run normally."
    )

enable_helmet = helmet_model is not None

model = load_vehicle_model(MODEL_SIZE)

# ---------------------------------------------------------------------------
# Input source selection
# ---------------------------------------------------------------------------
st.sidebar.header("Input Source")
camera_location = st.sidebar.text_input(
    "Camera / Location name",
    value="Camera 1",
    help="Tagged onto every logged violation — useful once you have more than one camera/site.",
)
input_source = st.sidebar.radio(
    "Choose input source",
    ["Upload Video", "CCTV / IP Camera"],
)

# --- Telegram alert settings ---
st.sidebar.header("📲 Telegram Alerts")

# If nothing was found in env vars / .env / secrets, let the user type the
# same bot token + chat ID they used in their other projects.
if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
    with st.sidebar.expander("Enter Telegram details manually", expanded=True):
        typed_token = st.text_input("Bot token", type="password", key="tg_token_input")
        typed_chat = st.text_input("Chat ID", key="tg_chat_input")
        if typed_token:
            TELEGRAM_BOT_TOKEN = typed_token.strip()
        if typed_chat:
            TELEGRAM_CHAT_ID = typed_chat.strip()

TELEGRAM_AVAILABLE = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and REQUESTS_AVAILABLE)

if TELEGRAM_AVAILABLE:
    st.sidebar.success("Telegram connected")
    if st.sidebar.button("📨 Send test message"):
        ok, detail = send_telegram_test()
        if ok:
            st.sidebar.success("Test message sent — check Telegram")
        else:
            st.sidebar.error(f"Failed: {detail}")
    telegram_mode = st.sidebar.selectbox(
        "Send violation images",
        [TELEGRAM_MODE_INSTANT, TELEGRAM_MODE_ACCEPTED, TELEGRAM_MODE_BOTH, TELEGRAM_MODE_OFF],
        help=(
            "Instant alerts may include false positives (the model isn't perfect). "
            "'After Accept' sends only human-confirmed violations."
        ),
    )
else:
    telegram_mode = TELEGRAM_MODE_OFF
    if not REQUESTS_AVAILABLE:
        st.sidebar.warning("Install `requests` (pip install requests) to enable Telegram alerts.")
    else:
        st.sidebar.warning("Telegram alerts OFF — enter your token and chat ID above.")

uploaded_file = None
cctv_url = None

if input_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

elif input_source == "CCTV / IP Camera":
    cctv_url = st.sidebar.text_input(
        "Stream URL (RTSP/HTTP)",
        placeholder="rtsp://username:password@192.168.1.10:554/stream1",
        help="Most IP cameras/CCTV/NVRs expose an RTSP URL. Check your camera's manual for the exact format.",
    )
    if cctv_url:
        st.info("CCTV stream URL set. Click **Play with detection** to connect and start streaming.")
    else:
        st.warning("Enter your CCTV/IP camera stream URL in the sidebar to continue.")

col_a, col_b = st.columns(2)
with col_a:
    start_btn = st.button("▶️ Play with detection", type="primary")
with col_b:
    stop_btn = st.button("⏹️ Stop")

if "running" not in st.session_state:
    st.session_state.running = False
if "review_status" not in st.session_state:
    st.session_state.review_status = {}
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if start_btn:
    st.session_state.running = True
    # Starting a fresh run — clear any previous run's results/review state
    # so old snapshots and accept/reject decisions don't linger.
    st.session_state.review_status = {}
    st.session_state.results_ready = False
if stop_btn:
    st.session_state.running = False

frame_slot = st.empty()
stats_slot = st.empty()
violation_slot = st.empty()


def find_associated_person(vx1, vy1, vx2, vy2, persons):
    """
    Given a two-wheeler's box and a list of PERSON boxes detected in the
    SAME frame, find the person most likely riding it. This is far more
    reliable than any geometric guess based on the vehicle box alone,
    because it's grounded in something the model actually saw, not an
    assumption about typical camera distance/angle.

    Primary method: highest IoU overlap (a rider's body usually overlaps
    the vehicle box significantly). Fallback: the person whose horizontal
    center falls within the vehicle's width, closest vertically above it
    (covers cases like a leaning rider or a very tight vehicle box).
    """
    if not persons:
        return None
    best, best_iou = None, 0.0
    for (px1, py1, px2, py2) in persons:
        ix1, iy1 = max(vx1, px1), max(vy1, py1)
        ix2, iy2 = min(vx2, px2), min(vy2, py2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = (vx2 - vx1) * (vy2 - vy1) + (px2 - px1) * (py2 - py1) - inter
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou, best = iou, (px1, py1, px2, py2)

    if best is None:
        candidates = [p for p in persons if vx1 <= (p[0] + p[2]) / 2 <= vx2]
        if candidates:
            best = min(candidates, key=lambda p: abs(p[1] - vy1))
    return best


def ensure_min_width(x1, y1, x2, y2, frame_w, frame_h, min_ratio=0.55):
    """
    Prevent unreasonably narrow crops/boxes — this was a real bug: for
    front-facing two-wheeler shots (rider facing the camera), the detected
    vehicle box is naturally narrow (you're seeing handlebar width, not the
    rider's actual shoulder width). Padding a percentage of an already-narrow
    width still produces a thin sliver that clips most of the rider out —
    which is exactly what caused clearly bare-headed riders to still read as
    "ACCEPT-HELMET": the classifier was working from a crop that barely
    contained any of the actual head.

    This widens the box (symmetrically, around its horizontal center) until
    its width is at least `min_ratio` of its height, whenever it's currently
    narrower than that — a reasonable minimum for a person's head/shoulders.
    """
    width = x2 - x1
    height = max(1, y2 - y1)
    min_width = height * min_ratio
    if width < min_width:
        center_x = (x1 + x2) / 2
        half = min_width / 2
        x1 = int(center_x - half)
        x2 = int(center_x + half)
    x1 = max(0, x1)
    x2 = min(frame_w - 1, x2)
    return x1, y1, x2, y2


def crop_head_region(frame, x1, y1, x2, y2, persons=None):
    """
    Crop the likely rider head/helmet region for a two-wheeler.

    PRIMARY method: if an actual PERSON box was detected for this rider,
    crop the top ~35% of THAT real box — a genuine head region grounded in
    what the model actually detected, not a guess. This is what actually
    fixes close-range reliability: a real person box scales correctly with
    distance because it's detected directly, whereas a geometric guess off
    the vehicle box's own height breaks down at extremes (very close OR
    very far).

    FALLBACK (only if no person box was found this frame): the previous
    geometric heuristic — search upward from the vehicle box's top edge,
    scaled down for close-range shots where the vehicle box itself already
    covers most of the frame.
    """
    h, w = frame.shape[:2]

    person_box = find_associated_person(x1, y1, x2, y2, persons) if persons else None
    if person_box is not None:
        px1, py1, px2, py2 = person_box
        person_h = py2 - py1
        # Small padding for context, then the top ~35-40% of the real person box
        pad_x = int((px2 - px1) * 0.10)
        head_y2 = int(py1 + person_h * 0.38)
        x1c, y1c = max(0, px1 - pad_x), max(0, py1)
        x2c, y2c = min(w, px2 + pad_x), min(h, max(head_y2, py1 + 1))
        x1c, y1c, x2c, y2c = ensure_min_width(x1c, y1c, x2c, y2c, w, h)
        if x2c > x1c and y2c > y1c:
            return frame[y1c:y2c, x1c:x2c]

    # Fallback: geometric heuristic (no person box available this frame)
    box_h = y2 - y1
    box_w = x2 - x1
    box_h_frac = box_h / max(1, h)
    if box_h_frac < 0.3:
        extension_multiplier = 1.0
    elif box_h_frac < 0.6:
        extension_multiplier = 0.5
    else:
        extension_multiplier = 0.15

    search_y1 = int(y1 - box_h * extension_multiplier)
    search_y2 = int(y1 + box_h * 0.35)
    pad_x = int(box_w * 0.15)
    search_x1 = x1 - pad_x
    search_x2 = x2 + pad_x

    x1c, y1c = max(0, search_x1), max(0, search_y1)
    x2c, y2c = min(w, search_x2), min(h, search_y2)
    x1c, y1c, x2c, y2c = ensure_min_width(x1c, y1c, x2c, y2c, w, h)
    if x2c <= x1c or y2c <= y1c:
        return frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]  # fallback to original box
    return frame[y1c:y2c, x1c:x2c]


MIN_CROP_DIM = 35          # crops smaller than this on either side are too little info to trust
MIN_SHARPNESS = 12.0        # Laplacian variance threshold — below this, the crop is likely too blurry


def is_crop_usable(crop) -> bool:
    """
    Quality gate: decide whether a head/rider crop is even worth running
    through the helmet classifier. Forcing a helmet/no-helmet guess on a
    tiny, dark, or badly blurred crop is a direct source of misclassification
    — it's better to treat the frame as INCONCLUSIVE (skip it, don't vote
    either way) than to guess on genuinely unreliable image data.
    """
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if min(h, w) < MIN_CROP_DIM:
        return False
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < MIN_SHARPNESS:
            return False
    except Exception:
        return False
    return True


def classify_helmet(head_crop):
    """
    Return (is_no_helmet, label, confidence):
      - is_no_helmet = True  -> confirmed NON-HELMET (confident violation signal)
      - is_no_helmet = False -> confirmed ACCEPT-HELMET (a REAL positive detection)
      - is_no_helmet = None  -> INCONCLUSIVE: the model detected NOTHING at all in
        this crop, or nothing crossed either confidence bar. This must NOT be
        silently treated as "helmet" — that was a real bug: a model finding
        zero relevant objects in a crop (wrong angle, occlusion, model just
        missing it) was being counted as positive evidence of a helmet, which
        is backwards. An inconclusive frame contributes NO vote either way.
    """
    try:
        h_results = helmet_model.predict(
            head_crop,
            conf=HELMET_INFERENCE_CONF,
            verbose=False,
            imgsz=640,
        )
        h_result = h_results[0]

        if h_result.boxes is None or len(h_result.boxes) == 0:
            return None, "NO-DETECTION", 0.0

        helmet_conf = 0.0
        no_helmet_conf = 0.0

        for det in h_result.boxes:
            conf = float(det.conf[0])
            cls_idx = int(det.cls[0])
            raw_name = helmet_model.names[cls_idx]
            name = str(raw_name).strip().lower()
            normalized = " ".join(
                name.replace("_", " ").replace("-", " ").replace("–", " ").replace("—", " ").split()
            )

            is_no_helmet = (
                "non helmet" in normalized
                or "nonhelmet" in normalized
                or "no helmet" in normalized
                or "without helmet" in normalized
                or "unhelmeted" in normalized
                or "un helmet" in normalized
            )

            is_helmet = (
                normalized == "helmet"
                or "accept helmet" in normalized
                or "with helmet" in normalized
                or "helmeted" in normalized
            )

            if is_no_helmet:
                no_helmet_conf = max(no_helmet_conf, conf)
            elif is_helmet:
                helmet_conf = max(helmet_conf, conf)

        # Explicit helmet detection wins when present — a REAL positive read.
        if helmet_conf >= HELMET_INFERENCE_CONF and helmet_conf >= no_helmet_conf:
            return False, "ACCEPT-HELMET", helmet_conf

        # NON-HELMET requires strong explicit evidence.
        if no_helmet_conf >= NO_HELMET_CONF:
            return True, "NON-HELMET", no_helmet_conf

        # Neither bar was clearly crossed — genuinely inconclusive, not "helmet".
        return None, "UNCLEAR", max(helmet_conf, no_helmet_conf)

    except Exception:
        return None, "ERROR", 0.0


def get_rider_display_box(frame, x1, y1, x2, y2, cls_id, persons=None):
    """
    Return a larger box for displaying two-wheelers, so the drawn box the
    person sees on screen matches what's actually being classified.

    PRIMARY method: if a real PERSON box was detected for this rider,
    union it with the vehicle box — a real, grounded region rather than a guess.

    FALLBACK: the previous geometric heuristic, scaled down for close-range
    shots where the vehicle box already fills most of the frame.
    """
    if cls_id not in TWO_WHEELER_IDS:
        return x1, y1, x2, y2

    h, w = frame.shape[:2]

    person_box = find_associated_person(x1, y1, x2, y2, persons) if persons else None
    if person_box is not None:
        px1, py1, px2, py2 = person_box
        dx1 = max(0, min(x1, px1))
        dy1 = max(0, min(y1, py1))
        dx2 = min(w - 1, max(x2, px2))
        dy2 = min(h - 1, max(y2, py2))
        return ensure_min_width(dx1, dy1, dx2, dy2, w, h)

    # Fallback: geometric heuristic (no person box available this frame)
    box_h = max(1, y2 - y1)
    box_w = max(1, x2 - x1)
    box_h_frac = box_h / max(1, h)
    if box_h_frac < 0.3:
        top_extra_multiplier = 1.15
    elif box_h_frac < 0.6:
        top_extra_multiplier = 0.55
    else:
        top_extra_multiplier = 0.15

    top_extra = int(box_h * top_extra_multiplier)
    side_pad = int(box_w * 0.25)
    bottom_pad = int(box_h * 0.12)

    dx1 = max(0, x1 - side_pad)
    dy1 = max(0, y1 - top_extra)
    dx2 = min(w - 1, x2 + side_pad)
    dy2 = min(h - 1, y2 + bottom_pad)

    return ensure_min_width(dx1, dy1, dx2, dy2, w, h)


# Determine whether we have a valid, ready-to-run source for the chosen input type
source_ready = (
    (input_source == "Upload Video" and uploaded_file is not None)
    or (input_source == "CCTV / IP Camera" and bool(cctv_url))
)

if source_ready and st.session_state.running:
    # Resolve the actual cv2.VideoCapture source based on the chosen input type
    if input_source == "Upload Video":
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)
    else:  # CCTV / IP Camera
        cap = cv2.VideoCapture(cctv_url)

    if not cap.isOpened():
        st.error(
            "Could not open this source. Double-check the CCTV/IP camera URL, "
            "credentials, and that the camera is reachable on your network."
        )
        st.session_state.running = False
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    # Live sources (CCTV) often report 0 or bogus FPS — fall back to a sane default
    if fps <= 1:
        fps = 25
    delay = 1.0 / fps

    frame_count = 0
    seen_ids = defaultdict(set)          # class_name -> set of unique track ids (for counting)
    track_history = defaultdict(lambda: deque(maxlen=8))  # track_id -> deque of (cx, cy, frame_idx)
    total_violations = set()             # track ids confirmed as "no helmet"
    helmet_votes = defaultdict(lambda: deque(maxlen=NO_HELMET_VOTE_WINDOW))
    last_no_helmet_conf = {}  # track_id -> highest no-helmet confidence seen, for logging
    track_second_opinion = {}  # track_id -> "HELMET" / "NOHELMET" / "UNCLEAR", consulted once per rider
    violation_snapshots = []             # list of (track_id, jpeg_bytes, violation_id, confidence, plate, opinion)

    # Set up an output video writer so the annotated result can be downloaded afterward.
    # Initialized lazily once we know the actual frame size.
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    video_writer = None

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Tracking (persist=True keeps IDs consistent across frames in this loop).
        # PERSON_CLASS_ID is included (when helmet checking is on) so we can
        # locate each rider's ACTUAL detected position, instead of guessing
        # it from the vehicle box's geometry alone — this is what makes
        # close-range and unusual-angle shots reliable.
        track_classes = list(VEHICLE_CLASSES.keys())
        if enable_helmet:
            track_classes = track_classes + [PERSON_CLASS_ID]

        # IMPORTANT FIX for close-range riders: when a rider gets very close
        # to the camera, the motorcycle/bicycle itself is often partially
        # cut off by the frame edge or motion-blurred — and at the normal
        # confidence threshold, that partial/blurred box gets filtered out
        # entirely. If the VEHICLE is never detected, the helmet check never
        # even runs for that rider, no matter how good the crop logic is.
        # Fix: run detection at a LOWER threshold so two-wheelers aren't lost
        # at exactly the range that matters most, then post-filter every
        # OTHER class back up to the original stricter bar so car/bus/truck
        # precision doesn't suffer.
        TWO_WHEELER_DETECT_CONF = min(0.15, conf_threshold)
        detection_conf = TWO_WHEELER_DETECT_CONF if enable_helmet else conf_threshold

        results = model.track(
            frame,
            conf=detection_conf,
            classes=track_classes,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        result = results[0]
        counts_this_frame = defaultdict(int)

        # First pass: collect every PERSON box detected in this frame. Used
        # only to locate riders for the helmet check — persons themselves
        # are not counted or drawn as vehicles. Persons get their own
        # reasonable confidence floor (not the ultra-low two-wheeler-only
        # threshold above) since a person is generally easy to detect at any
        # distance and a low bar here would just invite noisy false boxes.
        PERSON_MIN_CONF = 0.25
        persons_this_frame = []
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls[0]) == PERSON_CLASS_ID and float(box.conf[0]) >= PERSON_MIN_CONF:
                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                    persons_this_frame.append((px1, py1, px2, py2))

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == PERSON_CLASS_ID:
                    continue  # already collected above; not a vehicle to draw/count
                conf = float(box.conf[0])

                # Post-filter: non-two-wheeler classes must still clear the
                # ORIGINAL stricter threshold — only two-wheelers benefit
                # from the lowered detection floor above.
                if cls_id not in TWO_WHEELER_IDS and conf < conf_threshold:
                    continue

                label = VEHICLE_CLASSES.get(cls_id, "vehicle")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else None

                counts_this_frame[label] += 1
                if track_id is not None:
                    seen_ids[label].add(track_id)

                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                speed_text = ""

                # --- Speed estimation ---
                if enable_speed and track_id is not None:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    hist = track_history[track_id]
                    hist.append((cx, cy, frame_count))
                    if len(hist) >= 2:
                        x_old, y_old, f_old = hist[0]
                        x_new, y_new, f_new = hist[-1]
                        dt = (f_new - f_old) / fps
                        if dt > 0:
                            px_dist = ((x_new - x_old) ** 2 + (y_new - y_old) ** 2) ** 0.5
                            meters = px_dist * (meters_per_100px / 100.0)
                            speed_mps = meters / dt
                            speed_kmh = speed_mps * 3.6
                            speed_text = f" {speed_kmh:.0f} km/h"

                # --- Helmet check for two-wheelers (local model, strengthened) ---
                is_violation = False
                helmet_debug_label = None
                helmet_confidence = 0.0
                if enable_helmet and cls_id in TWO_WHEELER_IDS:
                    head_crop = crop_head_region(frame, x1, y1, x2, y2, persons=persons_this_frame)

                    # Quality gate: don't force a guess on a crop too small or
                    # too blurry to trust. An inconclusive frame contributes
                    # NO vote either way internally (this is what protects
                    # accuracy — not counting bad evidence). But the DISPLAYED
                    # label is always normalized to one of the two binary
                    # states below — the person never sees "NO-DETECTION" or
                    # "UNCLEAR" on screen, only ACCEPT-HELMET or NON-HELMET.
                    crop_usable = is_crop_usable(head_crop)
                    frame_no_helmet = None  # None = inconclusive, not a vote
                    if crop_usable:
                        frame_no_helmet, helmet_debug_label, helmet_confidence = classify_helmet(head_crop)

                    # Normalize the on-screen label: an inconclusive read
                    # (bad crop, no detection, or genuinely ambiguous) still
                    # displays as ACCEPT-HELMET — the same fail-safe default
                    # already used everywhere else in this pipeline — never
                    # a third, confusing label state.
                    if helmet_debug_label not in ("ACCEPT-HELMET", "NON-HELMET"):
                        helmet_debug_label = "ACCEPT-HELMET"

                    if track_id is not None and frame_no_helmet is not None:
                        votes = helmet_votes[track_id]
                        votes.append(frame_no_helmet)

                        if frame_no_helmet:
                            last_no_helmet_conf[track_id] = max(
                                last_no_helmet_conf.get(track_id, 0.0), helmet_confidence
                            )

                        # Consult the AI second opinion ONCE per rider (not every
                        # frame — keeps this fast/cheap), the first time their
                        # confidence lands in the genuinely uncertain band. This
                        # now actually INFLUENCES the decision, not just a label
                        # shown after the fact: if the second opinion disagrees
                        # and says HELMET, we require a much stronger, harder to
                        # reach local vote count before still confirming a
                        # violation — protecting against exactly the borderline
                        # misreads that cause false accusations.
                        if (
                            frame_no_helmet
                            and ANTHROPIC_AVAILABLE
                            and track_id not in track_second_opinion
                            and BORDERLINE_LOW <= helmet_confidence <= BORDERLINE_HIGH
                        ):
                            ok, crop_jpeg = cv2.imencode(".jpg", head_crop)
                            if ok:
                                opinion = ai_second_opinion(crop_jpeg.tobytes())
                                track_second_opinion[track_id] = opinion or "UNCLEAR"

                        opinion = track_second_opinion.get(track_id)
                        if opinion == "HELMET":
                            # AI disagrees with a borderline local read — demand
                            # near-unanimous local evidence before still flagging.
                            required_votes = max(NO_HELMET_MIN_VOTES + 2, len(votes))
                            is_violation = (
                                len(votes) >= required_votes
                                and sum(votes) >= required_votes
                            )
                        else:
                            # No second opinion, or it agrees/is unclear — use
                            # the normal vote requirement as before.
                            is_violation = (
                                len(votes) >= NO_HELMET_MIN_VOTES
                                and sum(votes) >= NO_HELMET_MIN_VOTES
                            )
                    else:
                        is_violation = False

                if is_violation:
                    color = RED
                elif enable_helmet and cls_id in TWO_WHEELER_IDS:
                    color = GREEN

                # IMPORTANT:
                # The COCO motorcycle/bicycle box can contain mostly the vehicle,
                # while the rider's head is above it.  Use an expanded display
                # box for two-wheelers so the helmet can be visually inspected.
                display_x1, display_y1, display_x2, display_y2 = get_rider_display_box(
                    frame, x1, y1, x2, y2, cls_id, persons=persons_this_frame
                )

                # Make helmet/violation boxes clearly visible.
                box_thickness = 5 if is_violation else 3
                cv2.rectangle(
                    frame,
                    (display_x1, display_y1),
                    (display_x2, display_y2),
                    color,
                    box_thickness
                )

                id_text = f"#{track_id} " if track_id is not None else ""
                helmet_tag = f" | {helmet_debug_label}" if helmet_debug_label else ""
                label_text = f"{id_text}{label} {conf:.2f}{speed_text}{helmet_tag}"

                font_scale = 0.65 if is_violation else 0.5
                font_thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                text_x, text_y = display_x1, max(display_y1 - 8, text_h + 5)
                # Filled background behind the label so it's readable over any footage,
                # and especially prominent (solid red) for violations
                bg_color = RED if is_violation else (0, 0, 0)
                cv2.rectangle(
                    frame,
                    (text_x, text_y - text_h - baseline - 2),
                    (text_x + text_w + 4, text_y + baseline),
                    bg_color,
                    -1,
                )
                text_color = (255, 255, 255)
                cv2.putText(
                    frame, label_text, (text_x + 2, text_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness,
                )

                # First time this rider is flagged — save a clearly-highlighted, zoomed-in
                # snapshot (captured AFTER the box/label above are drawn, so the highlight
                # actually appears in the saved image).
                if is_violation and track_id is not None and track_id not in total_violations:
                    h_frame, w_frame = frame.shape[:2]

                    # Base the review snapshot on the enlarged rider box too.
                    # This keeps the rider's head/helmet clearly visible.
                    snap_x1, snap_y1, snap_x2, snap_y2 = get_rider_display_box(
                        frame, x1, y1, x2, y2, cls_id
                    )

                    snap_box_w = snap_x2 - snap_x1
                    snap_box_h = snap_y2 - snap_y1
                    pad = int(max(snap_box_w, snap_box_h) * 0.20)

                    cx1 = max(0, snap_x1 - pad)
                    cy1 = max(0, snap_y1 - pad)
                    cx2 = min(w_frame, snap_x2 + pad)
                    cy2 = min(h_frame, snap_y2 + pad)

                    snap_crop = frame[cy1:cy2, cx1:cx2].copy()
                    if snap_crop.size > 0:
                        # Upscale small crops so the violation is clearly visible, not a tiny speck
                        crop_h, crop_w = snap_crop.shape[:2]
                        if crop_w < 500:
                            scale = 500 / crop_w
                            snap_crop = cv2.resize(snap_crop, (int(crop_w * scale), int(crop_h * scale)))
                        ok, jpeg_buf = cv2.imencode(".jpg", snap_crop)
                        if ok:
                            violation_id = str(uuid.uuid4())
                            confidence = last_no_helmet_conf.get(track_id, 0.0)
                            jpeg_bytes = jpeg_buf.tobytes()

                            # Best-effort local OCR for a license plate (fully offline,
                            # no external calls). None if unavailable/nothing found.
                            plate_number = read_plate_text(jpeg_bytes) if EASYOCR_AVAILABLE else None

                            # Reuse the second opinion already consulted (once per
                            # rider) during the live decision loop above, rather
                            # than calling the AI a second time for the same rider.
                            second_opinion = track_second_opinion.get(track_id)
                            if second_opinion is None and ANTHROPIC_AVAILABLE and BORDERLINE_LOW <= confidence <= BORDERLINE_HIGH:
                                second_opinion = ai_second_opinion(jpeg_bytes)

                            violation_snapshots.append(
                                (track_id, jpeg_bytes, violation_id, confidence, plate_number, second_opinion)
                            )
                            # Persistent audit log entry — survives across app
                            # restarts, unlike anything kept only in memory.
                            source_label = (
                                "Uploaded video" if input_source == "Upload Video" else (cctv_url or "CCTV")
                            )
                            log_violation(
                                violation_id, camera_location, source_label,
                                track_id, label, confidence,
                                plate_number=plate_number, ai_second_opinion=second_opinion,
                            )

                            # NEW: Telegram alert the moment a violation is first flagged
                            # (runs in a background thread; one message per rider).
                            if telegram_mode in (TELEGRAM_MODE_INSTANT, TELEGRAM_MODE_BOTH):
                                send_telegram_alert(
                                    jpeg_bytes, camera_location, label, track_id,
                                    confidence, plate_number=plate_number,
                                    status="🟡 Pending review",
                                )
                    total_violations.add(track_id)

        # Lazily initialize the output video writer using the first frame's actual size
        if video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        video_writer.write(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)

        stats_slot.write(
            f"**Frame {frame_count}** — this frame: "
            + " | ".join(f"{k}: {v}" for k, v in counts_this_frame.items())
            + "  \n**Unique counted so far:** "
            + " | ".join(f"{k}: {len(v)}" for k, v in seen_ids.items())
        )
        if enable_helmet:
            violation_slot.write(f"🔴 **Helmet violations (unique riders):** {len(total_violations)}")

        time.sleep(delay)

    cap.release()
    if video_writer is not None:
        video_writer.release()
    st.session_state.running = False

    # Persist everything needed to render results into session_state, since
    # clicking Accept/Reject below triggers a rerun — and this whole block
    # (gated on st.session_state.running) won't execute again after that,
    # so results must survive independently of it.
    st.session_state.results_ready = True
    st.session_state.results_input_source = input_source
    st.session_state.results_seen_ids = {k: len(v) for k, v in seen_ids.items()}
    st.session_state.results_enable_helmet = enable_helmet
    st.session_state.results_total_violations = len(total_violations)
    st.session_state.results_video_path = (
        output_video_path
        if (video_writer is not None and os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0)
        else None
    )
    st.session_state.violation_snapshots = violation_snapshots
    for vid_track_id, _, violation_id, _, _, _ in violation_snapshots:
        st.session_state.review_status.setdefault(violation_id, "pending")

elif not source_ready:
    if input_source == "Upload Video":
        st.info("Upload a video and click **Play with detection** to begin.")
    elif input_source == "CCTV / IP Camera":
        st.info("Enter a stream URL in the sidebar and click **Play with detection** to begin.")

# ---------------------------------------------------------------------------
# Results display — runs on every script execution (not just right after
# processing finishes), reading from session_state. This is what lets the
# Accept/Reject buttons below work: clicking one triggers a Streamlit rerun,
# and results need to still be there afterward.
# ---------------------------------------------------------------------------
if st.session_state.get("results_ready"):
    if st.session_state.results_input_source == "Upload Video":
        st.success("Video finished.")
    else:
        st.info("Stream stopped (either you clicked Stop, or the connection was lost).")

    st.write("### Summary")
    st.write("**Unique vehicles counted:**")
    st.write(st.session_state.results_seen_ids)
    if st.session_state.results_enable_helmet:
        st.write(f"**Total unique helmet violations:** {st.session_state.results_total_violations}")

    # --- Downloadable annotated video ---
    if st.session_state.results_video_path:
        with open(st.session_state.results_video_path, "rb") as f:
            video_bytes = f.read()
        st.download_button(
            "⬇️ Download annotated video (.mp4)",
            data=video_bytes,
            file_name="detected_output.mp4",
            mime="video/mp4",
            key="download_video_btn",
        )

    # --- Violation snapshots: human review (Accept/Reject) + filtered download ---
    if st.session_state.results_enable_helmet and st.session_state.violation_snapshots:
        st.write("### 📸 Helmet violation snapshots — review each one")
        st.caption(
            "The rider frame is intentionally enlarged upward so the head/helmet is visible. "
            "Automated detection isn't perfect — review each flagged rider before treating "
            "it as confirmed. Your decision updates the permanent audit log and is saved as "
            "a labeled training example for future model improvement."
        )

        cols = st.columns(3)
        for i, (vid_track_id, jpeg_bytes, violation_id, confidence, plate_number, second_opinion) in enumerate(
            st.session_state.violation_snapshots
        ):
            status = st.session_state.review_status.get(violation_id, "pending")
            status_label = {"pending": "🟡 Pending review", "accepted": "✅ Accepted", "rejected": "❌ Rejected"}[status]
            with cols[i % 3]:
                st.image(
                    jpeg_bytes,
                    caption=f"Rider #{vid_track_id} — {confidence*100:.0f}% confidence — {status_label}",
                    use_container_width=True,
                )
                if plate_number:
                    st.caption(f"🔢 Plate (OCR, unverified): {plate_number}")
                if second_opinion:
                    st.caption(f"🤖 AI second opinion (borderline case only): {second_opinion}")

                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button("✅ Accept", key=f"accept_{violation_id}", use_container_width=True):
                        st.session_state.review_status[violation_id] = "accepted"
                        update_review_status(violation_id, "accepted")
                        save_training_sample(jpeg_bytes, violation_id, accepted=True)

                        # NEW: Telegram alert for a human-CONFIRMED violation
                        if telegram_mode in (TELEGRAM_MODE_ACCEPTED, TELEGRAM_MODE_BOTH):
                            matching_row = next(
                                (r for r in fetch_all_violations() if r["violation_id"] == violation_id), None
                            )
                            send_telegram_alert(
                                jpeg_bytes,
                                matching_row["location"] if matching_row else camera_location,
                                matching_row["vehicle_type"] if matching_row else "motorcycle/scooter",
                                vid_track_id,
                                confidence,
                                plate_number=plate_number,
                                status="✅ Confirmed by reviewer",
                            )
                        st.rerun()
                with bcol2:
                    if st.button("❌ Reject", key=f"reject_{violation_id}", use_container_width=True):
                        st.session_state.review_status[violation_id] = "rejected"
                        update_review_status(violation_id, "rejected")
                        save_training_sample(jpeg_bytes, violation_id, accepted=False)
                        st.rerun()

                # Notice drafting only makes sense for a confirmed (accepted) violation —
                # the AI drafts the text, a human still reviews before it's ever sent.
                if status == "accepted" and ANTHROPIC_AVAILABLE:
                    if st.button("📝 Draft notice", key=f"draft_{violation_id}", use_container_width=True):
                        matching_row = next(
                            (r for r in fetch_all_violations() if r["violation_id"] == violation_id), None
                        )
                        if matching_row:
                            with st.spinner("Drafting..."):
                                notice_text = draft_violation_notice(matching_row)
                            st.text_area(
                                "Draft notice (review before using)",
                                value=notice_text or "Could not draft notice.",
                                key=f"notice_text_{violation_id}",
                                height=180,
                            )

        accepted_count = sum(1 for s in st.session_state.review_status.values() if s == "accepted")
        rejected_count = sum(1 for s in st.session_state.review_status.values() if s == "rejected")
        pending_count = sum(1 for s in st.session_state.review_status.values() if s == "pending")
        st.write(f"**Review status:** {accepted_count} accepted · {rejected_count} rejected · {pending_count} pending")

        accepted_snapshots = [
            (vid_track_id, jpeg_bytes, violation_id, confidence, plate_number, second_opinion)
            for vid_track_id, jpeg_bytes, violation_id, confidence, plate_number, second_opinion
            in st.session_state.violation_snapshots
            if st.session_state.review_status.get(violation_id) == "accepted"
        ]
        if accepted_snapshots:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for vid_track_id, jpeg_bytes, violation_id, confidence, plate_number, second_opinion in accepted_snapshots:
                    zf.writestr(f"violation_{violation_id}.jpg", jpeg_bytes)
            st.download_button(
                f"⬇️ Download ACCEPTED violation snapshots only ({len(accepted_snapshots)}) (.zip)",
                data=zip_buffer.getvalue(),
                file_name="helmet_violations_accepted.zip",
                mime="application/zip",
                key="download_accepted_zip_btn",
            )
        else:
            st.caption("No snapshots accepted yet — accept at least one above to enable the download.")

# ---------------------------------------------------------------------------
# Persistent violation log — reads from SQLite, so this shows the FULL
# history across every past run/session, not just what's in memory right
# now. This is the real-world audit trail: every violation ever flagged,
# when, where, how confident the model was, and what a human decided.
# ---------------------------------------------------------------------------
st.write("---")
with st.expander("📋 Full violation log (all sessions, persisted)"):
    all_rows = fetch_all_violations()
    if not all_rows:
        st.caption("No violations logged yet.")
    else:
        st.dataframe(all_rows, use_container_width=True, hide_index=True)

        total = len(all_rows)
        accepted = sum(1 for r in all_rows if r["review_status"] == "accepted")
        rejected = sum(1 for r in all_rows if r["review_status"] == "rejected")
        reviewed = accepted + rejected
        agreement_rate = (accepted / reviewed * 100) if reviewed > 0 else None

        st.write(f"**Total logged:** {total}  ·  **Reviewed:** {reviewed}  ·  **Pending:** {total - reviewed}")
        if agreement_rate is not None:
            st.write(
                f"**Your agreement rate with the model:** {agreement_rate:.0f}% "
                f"(of reviewed flags, {accepted} confirmed as real violations)"
            )
            st.caption(
                "Track this over time — a consistently high agreement rate is the "
                "real-world signal that the model is reliable enough to reduce manual review."
            )

        st.write("---")
        if not ANTHROPIC_AVAILABLE:
            st.caption("Install the `anthropic` package and set ANTHROPIC_API_KEY to enable AI reports/queries below.")
        else:
            st.write("#### 🤖 AI incident report")
            if st.button("Generate report", key="generate_report_btn"):
                with st.spinner("Generating..."):
                    report_text = generate_incident_report(all_rows)
                st.session_state["_last_report"] = report_text
            if st.session_state.get("_last_report"):
                st.write(st.session_state["_last_report"])

            st.write("#### 💬 Ask a question about this log")
            log_question = st.text_input(
                "e.g. 'How many violations happened at Camera 1?'", key="log_question_input"
            )
            if st.button("Ask", key="ask_log_btn") and log_question:
                with st.spinner("Thinking..."):
                    answer = answer_log_question(log_question, all_rows)
                st.write(answer or "Could not answer.")
