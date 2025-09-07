🎶 Emotion-Based Music Recommendation System

An AI-powered music recommender app built with Streamlit, OpenCV, TensorFlow, and Spotify API.
The system detects your facial emotion in real-time and recommends songs from Spotify playlists that match your mood.

✨ Features

🎭 Emotion Detection – Uses a CNN model trained on FER-2013 to detect emotions (Happy, Sad, Angry, Neutral, etc.)

🎶 Spotify Integration – Plays full songs directly in the app via Spotify’s Embed API

🤖 Smart Recommendations – Suggests songs based on detected emotions from a preprocessed dataset + Spotify search

📷 Live Webcam Support – Captures your mood in real-time using OpenCV

🎨 Modern UI – Dark-themed, card-style interface with interactive buttons

🛠️ Tech Stack

Frontend: Streamlit

Computer Vision: OpenCV

Deep Learning: TensorFlow / Keras

Music Data: Spotify Web API

Dataset: MUSE V3 + FER-2013

🚀 Installation

1️⃣ Clone this repo

git clone https://github.com/sakshirwt15/emotion-based-music-recommendation.git
cd emotion-based-music-recommendation


2️⃣ Install dependencies

pip install -r requirements.txt


3️⃣ Add Spotify credentials (create app at Spotify Dashboard
)

export SPOTIPY_CLIENT_ID="your_client_id"
export SPOTIPY_CLIENT_SECRET="your_client_secret"


(Or set them in .streamlit/secrets.toml for Streamlit Cloud.)

▶️ Run the App
streamlit run app.py


Then open: http://localhost:8501/

📸 How It Works

Start the webcam by clicking Start Emotion Scan

The CNN model detects your facial emotion

Recommended Spotify songs appear in cards with embedded player


MIT License © 2025 Sakshi rawat