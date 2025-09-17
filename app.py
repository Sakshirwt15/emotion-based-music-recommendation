"""
Emotion-based Music Recommendation System
- Detects emotion from webcam (CNN via streamlit-webrtc)
- Picks songs from CSV by emotion
- Looks up each song on Spotify
- Plays tracks with embedded Spotify player
"""

import os
import pathlib
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import cv2

# New import for webcam streaming
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Keras (via TensorFlow)
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D  # type: ignore

# Spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ------------------------- CONFIG ------------------------- #
BASE_DIR = pathlib.Path(__file__).parent

st.set_page_config(
    page_title="Emotion Music Recommender",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence TF logs

# ------------------------- STYLE -------------------------- #
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 38px;
            font-weight: 700;
            color: #1DB954;
            margin-bottom: 6px;
        }
        .sub-title {
            text-align: center;
            font-size: 16px;
            color: #bdbdbd;
            margin-bottom: 22px;
        }
        .emotion-box {
            background-color: transparent;
            padding: 12px;
            text-align: center;
            font-size: 20px;
            font-weight: 700;
            color: white;
            margin: 8px 0 14px 0;
        }
        .song-card {
            background-color: white;
            padding: 12px 16px;
            border-radius: 12px;
            margin: 10px 0 18px 0;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
        }
        .song-title {
            color: #000;
            font-weight: 700;
            font-size: 18px;
        }
        .song-artist {
            color: #444;
            font-style: italic;
            font-size: 16px;
        }
        .footer {
            text-align: center;
            margin-top: 32px;
            font-size: 13px;
            color: gray;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------- SPOTIFY AUTH ----------------------- #
CLIENT_ID = "16cc572754584e54bfcbe88fff303ba1"
CLIENT_SECRET = "bee35803bf21489ba44dbf6a2ded4b7e"

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET
    )
)


@st.cache_data(show_spinner=False)
def spotify_search(track_name: str, artist_name: str):
    try:
        query = f"track:{track_name} artist:{artist_name}"
        res = sp.search(q=query, type="track", limit=1)
        items = res.get("tracks", {}).get("items", [])
        if items:
            t = items[0]
            return {
                "id": t["id"],
                "url": t["external_urls"]["spotify"],
                "name": t["name"],
                "artist": ", ".join(a["name"] for a in t["artists"]),
            }
    except Exception:
        pass
    return None


def spotify_rows_from_df(df_in: pd.DataFrame, limit: int = 10):
    rows = []
    for _, r in df_in.head(limit).iterrows():
        info = spotify_search(str(r["name"]), str(r["artist"]))
        rows.append(
            {
                "name": str(r["name"]),
                "artist": str(r["artist"]),
                "spotify_id": info["id"] if info else None,
                "spotify_url": info["url"] if info else None,
                "fallback_url": str(r["link"]),
            }
        )
    return rows


# ---------------------- LOAD DATA ------------------------- #
csv_path = BASE_DIR / "muse_v3.csv"
model_path = BASE_DIR / "model.h5"
haar_path = BASE_DIR / "haarcascade_frontalface_default.xml"

if not os.path.exists(csv_path):
    st.error(f"CSV file not found at {csv_path}")
    st.stop()
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()
if not os.path.exists(haar_path):
    st.error(f"Haarcascade file not found at {haar_path}")
    st.stop()

df = pd.read_csv(csv_path)
df["link"] = df["lastfm_url"]
df["name"] = df["track"]
df["emotional"] = df["number_of_emotion_tags"]
df["pleasant"] = df["valence_tags"]
df = df[["name", "emotional", "pleasant", "link", "artist"]]
df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

# Create emotion slices
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]


def recommend_songs(emotion_list):
    data = pd.DataFrame()
    times_dict = {
        1: [30],
        2: [30, 20],
        3: [55, 20, 15],
        4: [30, 29, 18, 9],
        5: [10, 7, 6, 5, 2],
    }
    t_list = times_dict.get(len(emotion_list), [10, 7, 6, 5, 2])

    for i, v in enumerate(emotion_list):
        t = t_list[i]
        try:
            if v == "Neutral":
                data = pd.concat([data, df_neutral.sample(n=min(t, len(df_neutral)))], ignore_index=True)
            elif v == "Angry":
                data = pd.concat([data, df_angry.sample(n=min(t, len(df_angry)))], ignore_index=True)
            elif v == "Fearful":
                data = pd.concat([data, df_fear.sample(n=min(t, len(df_fear)))], ignore_index=True)
            elif v == "Happy":
                data = pd.concat([data, df_happy.sample(n=min(t, len(df_happy)))], ignore_index=True)
            else:
                data = pd.concat([data, df_sad.sample(n=min(t, len(df_sad)))], ignore_index=True)
        except ValueError:
            continue
    return data


# ---------------------- EMOTION MODEL --------------------- #
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))


model.load_weights(model_path)

emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

face_cascade = cv2.CascadeClassifier(str(haar_path))

# ---------------------- STREAMLIT UI ---------------------- #
st.markdown("<div class='main-title'>🎶 Emotion-Based Music Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Detect your mood and instantly get songs you can play on Spotify</div>", unsafe_allow_html=True)

if "emotions_detected" not in st.session_state:
    st.session_state["emotions_detected"] = []


# --------------- Emotion Scan (via streamlit-webrtc) ------ #
class EmotionDetector(VideoTransformerBase):
    def __init__(self):
        self.emotions = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48)).astype("float32") / 255.0
            cropped_img = np.expand_dims(np.expand_dims(roi_gray, -1), 0)

            pred = model.predict(cropped_img, verbose=0)
            max_index = int(np.argmax(pred))
            emotion = emotion_dict[max_index]
            self.emotions.append(emotion)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        return img


st.write("📸 Start the webcam below to detect your emotion:")

webrtc_ctx = webrtc_streamer(
    key="emotion-detect",
    video_transformer_factory=EmotionDetector,
)

if webrtc_ctx and webrtc_ctx.video_transformer:
    if st.button("✅ Save Final Emotion"):
        if webrtc_ctx.video_transformer.emotions:
            final_emotion = Counter(webrtc_ctx.video_transformer.emotions).most_common(1)[0][0]
            st.session_state["emotions_detected"] = [final_emotion]
            st.success(f"Final Detected Emotion: {final_emotion}")
        else:
            st.warning("⚠️ No face detected. Try again with better lighting.")


# --------------- Recommendations + Spotify --------------- #
if st.session_state["emotions_detected"]:
    st.markdown(
        f"<div class='emotion-box'>Detected Emotions: {', '.join(st.session_state['emotions_detected'])}</div>",
        unsafe_allow_html=True,
    )

    if st.button("🎵 Recommend & Play Songs"):
        rec_df = recommend_songs(st.session_state["emotions_detected"])
        st.markdown("<h3 style='text-align:center;'>✨ Recommended Songs for You</h3>", unsafe_allow_html=True)

        rows = spotify_rows_from_df(rec_df, limit=10)

        for i, s in enumerate(rows, start=1):
            st.markdown(
                f"""
                <div class="song-card">
                    <div class="song-title">{i}. {s['name']}</div>
                    <div class="song-artist">{s['artist']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if s["spotify_id"]:
                components.iframe(
                    src=f"https://open.spotify.com/embed/track/{s['spotify_id']}?utm_source=generator",
                    height=80,
                    width=700,
                    scrolling=False,
                )
            else:
                st.markdown(f"[🔗 Open (fallback link)]({s['fallback_url']})", unsafe_allow_html=True)


# ------------------------- Footer ------------------------ #
st.markdown("<div class='footer'>Made with ❤️ using Streamlit, OpenCV, TensorFlow & Spotify API</div>", unsafe_allow_html=True)
