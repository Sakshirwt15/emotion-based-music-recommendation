# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

# ---------- Load CSV safely ----------
csv_path = r"C:\Users\Shakshi Rawat\Downloads\muse_v3.csv"  # Update if different
if not os.path.exists(csv_path):
    st.error(f"CSV file not found at {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)

# Standardize column names
df["link"] = df["lastfm_url"]
df["name"] = df["track"]
df["emotional"] = df["number_of_emotion_tags"]
df["pleasant"] = df["valence_tags"]

df = df[["name", "emotional", "pleasant", "link", "artist"]]
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index(drop=True, inplace=True)

# Split dataset
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]


# ---------- Functions ----------
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
                data = pd.concat(
                    [data, df_neutral.sample(n=min(t, len(df_neutral)))],
                    ignore_index=True,
                )
            elif v == "Angry":
                data = pd.concat(
                    [data, df_angry.sample(n=min(t, len(df_angry)))], ignore_index=True
                )
            elif v == "Fearful":
                data = pd.concat(
                    [data, df_fear.sample(n=min(t, len(df_fear)))], ignore_index=True
                )
            elif v == "Happy":
                data = pd.concat(
                    [data, df_happy.sample(n=min(t, len(df_happy)))], ignore_index=True
                )
            else:  # Sad
                data = pd.concat(
                    [data, df_sad.sample(n=min(t, len(df_sad)))], ignore_index=True
                )
        except ValueError:
            continue  # in case dataset is too small

    return data


def preprocess_emotions(l):
    """Remove duplicates but keep frequency count"""
    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)

    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
    return ul


# ---------- Load Model ----------
model_path = r"C:\Users\Shakshi Rawat\Downloads\model.h5"  # Update if different
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()

model = Sequential(
    [
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation="relu"),
        Dropout(0.5),
        Dense(7, activation="softmax"),
    ]
)
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

# ---------- Load Haarcascade ----------
haar_path = r"C:\Users\Shakshi Rawat\Downloads\haarcascade_frontalface_default.xml"  # Update if different
face_cascade = cv2.CascadeClassifier(haar_path)
if face_cascade.empty():
    st.error("Haarcascade XML failed to load.")
    st.stop()

# ---------- Streamlit UI ----------
page_bg_img = """
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown(
    "<h2 style='text-align: center; color: white'><b>Emotion based music recommendation</b></h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
list_emotions = []

with col2:
    if st.button("SCAN EMOTION (Click here)"):
        cap = cv2.VideoCapture(0)
        count = 0
        list_emotions.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            count += 1

            for x, y, w, h in faces:
                roi_gray = gray[y : y + h, x : x + w]
                cropped_img = np.expand_dims(
                    np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
                )
                prediction = model.predict(cropped_img, verbose=0)
                max_index = int(np.argmax(prediction))
                list_emotions.append(emotion_dict[max_index])

                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    emotion_dict[max_index],
                    (x + 20, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Video", cv2.resize(frame, (800, 600)))

            if cv2.waitKey(1) & 0xFF == ord("s") or count >= 20:
                break

        cap.release()
        cv2.destroyAllWindows()
        list_emotions = preprocess_emotions(list_emotions)
        st.success("Emotions successfully detected")

# ---------- Recommend Songs ----------
if list_emotions:
    new_df = recommend_songs(list_emotions)

    st.markdown(
        "<h5 style='text-align: center; color: grey;'><b>Recommended songs with artist names</b></h5>",
        unsafe_allow_html=True,
    )
    st.write(
        "---------------------------------------------------------------------------------------------------------------------"
    )

    try:
        for i, row in new_df.head(30).iterrows():
            st.markdown(
                f"<h4 style='text-align: center;'><a href='{row['link']}' target='_blank'>{i+1}. {row['name']}</a></h4>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h5 style='text-align: center; color: grey;'><i>{row['artist']}</i></h5>",
                unsafe_allow_html=True,
            )
            st.write(
                "---------------------------------------------------------------------------------------------------------------------"
            )
    except Exception as e:
        st.warning(f"Error displaying songs: {e}")
