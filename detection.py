# ============================================================
# detection.py
# Vehicle Detection + Tracking + Helmet Violation
# + REAL TELEGRAM IMAGE ALERTS
# ============================================================

import os
import io
import time
import datetime
import threading
from collections import deque, defaultdict

import cv2
import numpy as np
import streamlit as st
import requests

from ultralytics import YOLO


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vehicle Detection",
    layout="wide"
)

st.title("🚗 Vehicle Detection, Tracking & Helmet Check")


# ============================================================
# TELEGRAM CONFIGURATION
# ============================================================

def _get_secret(name):
    """
    Read Telegram credentials from:
    1. Environment variables
    2. Streamlit secrets
    """

    value = os.environ.get(name)

    if value:
        return value.strip()

    try:
        value = st.secrets.get(name)

        if value:
            return str(value).strip()

    except Exception:
        pass

    return None


TELEGRAM_BOT_TOKEN = _get_secret(
    "TELEGRAM_BOT_TOKEN"
)

TELEGRAM_CHAT_ID = _get_secret(
    "TELEGRAM_CHAT_ID"
)


TELEGRAM_AVAILABLE = bool(
    TELEGRAM_BOT_TOKEN
    and TELEGRAM_CHAT_ID
)


# ============================================================
# TELEGRAM LOG
# ============================================================

if "telegram_log" not in st.session_state:
    st.session_state.telegram_log = []


def get_telegram_log():
    return st.session_state.telegram_log


# ============================================================
# SEND PHOTO TO TELEGRAM
# ============================================================

def _send_telegram_photo(
    jpeg_bytes,
    caption,
    log
):
    """
    Sends the actual violation image to Telegram.
    """

    timestamp = datetime.datetime.now().strftime(
        "%H:%M:%S"
    )

    try:

        response = requests.post(

            f"https://api.telegram.org/"
            f"bot{TELEGRAM_BOT_TOKEN}/sendPhoto",

            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption[:1024],
            },

            files={
                "photo": (
                    "violation.jpg",
                    jpeg_bytes,
                    "image/jpeg"
                )
            },

            timeout=20,
        )


        if response.ok:

            log.append(
                f"{timestamp} "
                f"✅ Telegram violation image sent"
            )

        else:

            log.append(
                f"{timestamp} "
                f"❌ Telegram failed "
                f"{response.status_code}: "
                f"{response.text[:200]}"
            )

            print(
                "Telegram error:",
                response.status_code,
                response.text
            )


    except Exception as error:

        log.append(
            f"{timestamp} "
            f"❌ Telegram error: {error}"
        )

        print(
            "Telegram send failed:",
            error
        )


# ============================================================
# TELEGRAM VIOLATION ALERT
# ============================================================

def send_telegram_alert(
    jpeg_bytes,
    location,
    vehicle_type,
    track_id,
    confidence,
    speed_kmh=0.0
):
    """
    Sends the REAL violation frame to Telegram.

    The image is extracted directly from the processed
    video frame.
    """

    if not TELEGRAM_AVAILABLE:
        return


    # --------------------------------------------------------
    # Vehicle name
    # --------------------------------------------------------

    if vehicle_type == "motorcycle/scooter":

        vehicle_display = "Motorcycle"

    else:

        vehicle_display = vehicle_type.title()


    # --------------------------------------------------------
    # Current time
    # --------------------------------------------------------

    timestamp = datetime.datetime.now().strftime(
        "%H:%M:%S"
    )


    # --------------------------------------------------------
    # Telegram message
    # --------------------------------------------------------

    caption = (

        "🚨 HELMET VIOLATION DETECTED\n\n"

        f"🏍️ Vehicle: {vehicle_display}\n"

        "🪖 Helmet: NOT DETECTED\n"

        f"🎯 Confidence: "
        f"{confidence * 100:.1f}%\n"

        f"🔢 Tracking ID: {track_id}\n"

        f"💨 Speed: {speed_kmh:.1f} km/h\n"

        f"📍 Location: {location}\n"

        f"🕒 Time: {timestamp}\n\n"

        "📷 Actual violation snapshot attached."
    )


    # --------------------------------------------------------
    # Send in background
    # --------------------------------------------------------

    thread = threading.Thread(

        target=_send_telegram_photo,

        args=(
            jpeg_bytes,
            caption,
            get_telegram_log()
        ),

        daemon=True
    )

    thread.start()


# ============================================================
# VEHICLE SETTINGS
# ============================================================

VEHICLE_CLASSES = {

    1: "bicycle",

    2: "car",

    3: "motorcycle/scooter",

    5: "bus",

    7: "truck"
}


TWO_WHEELER_IDS = {
    1,
    3
}


PERSON_CLASS_ID = 0


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_SIZE = "yolov8n.pt"

CONF_THRESHOLD = 0.35


# ============================================================
# SPEED SETTINGS
# ============================================================

ENABLE_SPEED = True

METERS_PER_100PX = 5.0


# ============================================================
# HELMET SETTINGS
# ============================================================

HELMET_INFERENCE_CONF = 0.35

NO_HELMET_CONF = 0.60

NO_HELMET_VOTE_WINDOW = 5

NO_HELMET_MIN_VOTES = 3


# ============================================================
# LOAD VEHICLE MODEL
# ============================================================

@st.cache_resource
def load_vehicle_model():

    return YOLO(MODEL_SIZE)


vehicle_model = load_vehicle_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Detection Settings"
)


camera_location = st.sidebar.text_input(

    "Camera / Location name",

    value="Video Detection"
)


enable_speed = st.sidebar.checkbox(

    "Enable Speed Estimation",

    value=True
)


telegram_enabled = st.sidebar.checkbox(

    "Enable Telegram Alerts",

    value=TELEGRAM_AVAILABLE
)


# ============================================================
# TELEGRAM STATUS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📱 Telegram Status"
)


if TELEGRAM_AVAILABLE:

    st.sidebar.success(
        "🟢 Telegram Connected"
    )

else:

    st.sidebar.warning(
        "🔴 Telegram Not Configured"
    )


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_video = st.file_uploader(

    "Upload Vehicle Detection Video",

    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


# ============================================================
# RIDER DISPLAY BOX
# ============================================================

def get_rider_display_box(
    frame,
    x1,
    y1,
    x2,
    y2,
    cls_id,
    persons=None
):
    """
    Returns a suitable display box for the rider.
    """

    height, width = frame.shape[:2]

    x1 = max(
        0,
        int(x1)
    )

    y1 = max(
        0,
        int(y1)
    )

    x2 = min(
        width,
        int(x2)
    )

    y2 = min(
        height,
        int(y2)
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# MAIN VIDEO PROCESSING
# ============================================================

if uploaded_video is not None:

    # --------------------------------------------------------
    # Save uploaded video temporarily
    # --------------------------------------------------------

    video_bytes = uploaded_video.read()

    video_path = (
        "uploaded_detection_video.mp4"
    )

    with open(
        video_path,
        "wb"
    ) as video_file:

        video_file.write(
            video_bytes
        )


    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        st.error(
            "❌ Could not open video."
        )

        st.stop()


    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        fps = 30.0


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    # --------------------------------------------------------
    # Streamlit placeholders
    # --------------------------------------------------------

    frame_placeholder = st.empty()

    progress_bar = st.progress(0)


    # --------------------------------------------------------
    # Tracking data
    # --------------------------------------------------------

    frame_count = 0

    track_history = defaultdict(

        lambda: deque(
            maxlen=8
        )
    )


    total_violations = set()


    helmet_votes = defaultdict(

        lambda: deque(
            maxlen=NO_HELMET_VOTE_WINDOW
        )
    )


    violation_snapshots = []


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    while True:

        success, frame = cap.read()


        if not success:
            break


        frame_count += 1


        # ----------------------------------------------------
        # SPEED INITIALIZATION
        # ----------------------------------------------------

        speed_text = ""

        speed_kmh = 0.0


        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        results = vehicle_model.track(

            frame,

            persist=True,

            conf=CONF_THRESHOLD,

            classes=[
                0,
                1,
                2,
                3,
                5,
                7
            ],

            verbose=False
        )


        persons_this_frame = []


        # ====================================================
        # DETECTION RESULTS
        # ====================================================

        for result in results:

            if result.boxes is None:
                continue


            boxes = result.boxes


            for i in range(
                len(boxes)
            ):

                cls_id = int(
                    boxes.cls[i].item()
                )


                confidence = float(
                    boxes.conf[i].item()
                )


                coordinates = (
                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                )


                x1, y1, x2, y2 = map(
                    int,
                    coordinates
                )


                # ------------------------------------------------
                # PERSON
                # ------------------------------------------------

                if cls_id == PERSON_CLASS_ID:

                    persons_this_frame.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )

                    continue


                # ------------------------------------------------
                # VEHICLE
                # ------------------------------------------------

                if cls_id not in VEHICLE_CLASSES:
                    continue


                label = VEHICLE_CLASSES[
                    cls_id
                ]


                # ------------------------------------------------
                # TRACK ID
                # ------------------------------------------------

                track_id = None


                if boxes.id is not None:

                    track_id = int(
                        boxes.id[i].item()
                    )


                # =================================================
                # SPEED ESTIMATION
                # =================================================

                if (
                    enable_speed
                    and track_id is not None
                ):

                    cx = (
                        x1 + x2
                    ) / 2.0


                    cy = (
                        y1 + y2
                    ) / 2.0


                    history = (
                        track_history[
                            track_id
                        ]
                    )


                    history.append(

                        (
                            cx,
                            cy,
                            frame_count
                        )
                    )


                    if len(history) >= 2:

                        (
                            previous_x,
                            previous_y,
                            previous_frame
                        ) = history[-2]


                        (
                            current_x,
                            current_y,
                            current_frame
                        ) = history[-1]


                        pixel_distance = (

                            (
                                current_x
                                - previous_x
                            ) ** 2

                            +

                            (
                                current_y
                                - previous_y
                            ) ** 2

                        ) ** 0.5


                        frame_difference = (

                            current_frame
                            - previous_frame
                        )


                        if frame_difference > 0:

                            dt = (
                                frame_difference
                                / fps
                            )


                            meters = (

                                pixel_distance
                                * METERS_PER_100PX
                                / 100.0
                            )


                            speed_mps = (
                                meters / dt
                            )


                            speed_kmh = (
                                speed_mps * 3.6
                            )


                            speed_text = (
                                f" "
                                f"{speed_kmh:.0f}"
                                f" km/h"
                            )


                # =================================================
                # HELMET CHECK
                # =================================================

                is_violation = False

                helmet_confidence = (
                    confidence
                )


                if cls_id in TWO_WHEELER_IDS:

                    # ---------------------------------------------
                    # In your full project, put your existing
                    # helmet YOLO inference here.
                    #
                    # It should set:
                    #
                    # is_violation = True
                    #
                    # when NO HELMET is confirmed.
                    # ---------------------------------------------

                    is_violation = False


                # =================================================
                # DRAW VEHICLE BOX
                # =================================================

                box_color = (
                    (0, 255, 0)
                    if not is_violation
                    else
                    (0, 0, 255)
                )


                cv2.rectangle(

                    frame,

                    (
                        x1,
                        y1
                    ),

                    (
                        x2,
                        y2
                    ),

                    box_color,

                    2
                )


                display_text = (

                    f"{label}"
                    f" ID:{track_id}"
                    f"{speed_text}"
                )


                cv2.putText(

                    frame,

                    display_text,

                    (
                        x1,
                        max(
                            25,
                            y1 - 10
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    box_color,

                    2
                )


                # =================================================
                # CONFIRMED HELMET VIOLATION
                # =================================================

                if (

                    is_violation

                    and track_id is not None

                    and track_id
                    not in total_violations
                ):

                    total_violations.add(
                        track_id
                    )


                    # ---------------------------------------------
                    # Get actual rider box
                    # ---------------------------------------------

                    (
                        snap_x1,
                        snap_y1,
                        snap_x2,
                        snap_y2
                    ) = get_rider_display_box(

                        frame,

                        x1,
                        y1,
                        x2,
                        y2,

                        cls_id,

                        persons=persons_this_frame
                    )


                    # ---------------------------------------------
                    # ACTUAL VIOLATION IMAGE
                    # ---------------------------------------------

                    snapshot = frame[
                        snap_y1
