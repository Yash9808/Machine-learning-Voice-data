import os
import pandas as pd
import numpy as np
import whisper
import torchaudio
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from ftplib import FTP

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Streamlit App
st.title("📊 Telecall Analysis with FTP Audio Processing")

# FTP Login Sidebar
st.sidebar.header("📡 FTP Login")
host = st.sidebar.text_input("Host", "cph.v4one.co.uk")
username = st.sidebar.text_input("Username", "your_username")
password = st.sidebar.text_input("Password", type="password")

# Connect & List Directories
if st.sidebar.button("🔄 Connect & List Folders"):
    try:
        ftp = FTP(host, timeout=120)
        ftp.login(user=username, passwd=password)

        # List available directories
        st.write("📂 Available Directories on FTP:")
        folders = []
        ftp.retrlines("LIST", lambda x: (folders.append(x.split()[-1]), st.write(x)))

        available_dates = [folder for folder in folders if folder.startswith("2025")]

        ftp.quit()
        st.session_state["available_dates"] = available_dates
        st.success("✅ Connected! Select a date below.")
    except Exception as e:
        st.error(f"Connection failed: {e}")

# Select Date & Download Files
if "available_dates" in st.session_state:
    selected_date = st.selectbox("📅 Select a Date", st.session_state["available_dates"])

    if st.button("📥 Download & Process Audio"):
        try:
            ftp = FTP(host)
            ftp.login(user=username, passwd=password)

            # Set up paths
            remote_folder = selected_date
            local_folder = f"audio_files/{selected_date}"
            os.makedirs(local_folder, exist_ok=True)

            # Get list of audio files
            audio_files = []
            ftp.cwd(remote_folder)
            ftp.retrlines("LIST", lambda x: audio_files.append(x.split()[-1]))

            # Download files
            for file in audio_files:
                local_file_path = os.path.join(local_folder, file)
                with open(local_file_path, "wb") as f:
                    ftp.retrbinary(f"RETR {file}", f.write)

            ftp.quit()
            st.success(f"✅ Downloaded {len(audio_files)} files from {selected_date}")

            st.session_state["input_folder"] = local_folder
        except Exception as e:
            st.error(f"Download failed: {e}")

# Set input folder (After successful download)
if "input_folder" in st.session_state:
    INPUT_FOLDER = st.session_state["input_folder"]
else:
    INPUT_FOLDER = "audio_files"

# Load AI Model (Whisper for Transcription)
whisper_model = whisper.load_model("base")

# Define CSV Output
OUTPUT_CSV = "telecall_analysis.csv"

# Keywords for Processing
medical_tests = ["MRI", "X-Ray", "Ultrasound", "Endoscopy", "Gynaecology", "Orthopaedics", "General Surgery", "ENT"]
subscriptions = ["Gold", "Platinum", "Silver", "Bronze"]
upselling_phrases = [
    "You might want to consider", "A better option would be", "I recommend upgrading to",
    "You could save more by", "This plan offers better benefits", "Would you like to try our premium plan?"
]

# Load MP3 audio directly using torchaudio (No FFmpeg required)
def load_audio(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path, format="mp3")
        return waveform.squeeze(0).numpy(), sample_rate
    except Exception as e:
        st.error(f"❌ Failed to load MP3 audio: {e}")
        return None, None

# Process Audio Files
@st.cache_data
def process_audio_files():
    if not os.path.exists(INPUT_FOLDER):
        st.error(f"❌ Input folder not found: {INPUT_FOLDER}")
        st.stop()

    mp3_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".mp3")]

    if not mp3_files:
        st.error(f"❌ No MP3 files found in {INPUT_FOLDER}")
        st.stop()

    data = []

    for file in mp3_files:
        file_path = os.path.join(INPUT_FOLDER, file)

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            st.warning(f"⚠️ Skipping {file}: File is empty or missing.")
            continue

        try:
            # Load MP3 directly
            waveform, sample_rate = load_audio(file_path)
            if waveform is None:
                continue  # Skip if loading failed

            audio_length = len(waveform) / sample_rate

            # Transcription with Whisper
            result = whisper_model.transcribe(file_path)
            text = result.get("text", "")

            if not text.strip():
                st.warning(f"⚠️ Skipping {file}: No transcribed text detected.")
                continue

        except Exception as e:
            st.error(f"❌ Processing failed on {file}: {e}")
            continue  # Skip file

        # Named Entity Recognition (NER) using NLTK
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        entities = [word for word, tag in tagged_words if tag in ["NNP", "NN"]]

        # Extract Features
        agent_name, person_name, subscription, medical_test, age = "", "", "", "", ""
        emergency, problem, report_delay, upselling = "No", "No", "No", "No"

        for word in entities:
            if not agent_name:
                agent_name = word
            else:
                person_name = word

        if any(sub.lower() in text.lower() for sub in subscriptions):
            subscription = next(sub for sub in subscriptions if sub.lower() in text.lower())
        if any(test.lower() in text.lower() for test in medical_tests):
            medical_test = next(test for test in medical_tests if test.lower() in text.lower())
        if "urgent" in text.lower() or "immediate" in text.lower():
            emergency = "Yes"
        if "pain" in text.lower() or "issue" in text.lower():
            problem = "Yes"
        if "not received" in text.lower() or "waiting for report" in text.lower():
            report_delay = "Yes"

        # Sentiment Analysis
        sentiment_score = TextBlob(text).sentiment.polarity
        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

        # Upselling Detection
        if any(phrase.lower() in text.lower() for phrase in upselling_phrases):
            upselling = "Yes" if sentiment in ["Positive", "Neutral"] else "No"

        data.append([file, round(audio_length, 2), agent_name, person_name, subscription, medical_test,
                     age, problem, emergency, problem, report_delay, upselling, entities, sentiment])

    df = pd.DataFrame(data, columns=["File", "Audio Length (sec)", "Agent Name", "Person Name", "Subscription",
                                     "Medical Test", "Age", "Issue", "Emergency", "Problem", "Report Delay",
                                     "Upselling", "NER", "Sentiment"])
    df.to_csv(OUTPUT_CSV, index=False)
    return df

# Run Analysis
if "input_folder" in st.session_state:
    df = process_audio_files()
    st.write("## Processed Data")
    st.dataframe(df.head())

    st.success("✅ Analysis Complete!")
