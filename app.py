import streamlit as st
import os
import pandas as pd
import numpy as np
import librosa
import whisper
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import defaultdict
from ftplib import FTP

# Streamlit UI
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

        # List available directories (assuming they are date-based)
        st.write("📂 Available Directories on FTP:")
        folders = []
        ftp.retrlines("LIST", lambda x: (folders.append(x.split()[-1]), st.write(x)))

        available_dates = [folder for folder in folders if folder.startswith("2025")]  # Example for date-based folders

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

            # Save selected folder for processing
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

# Define Keywords for Processing
medical_tests = ["MRI", "X-Ray", "Ultrasound", "Endoscopy", "Gynaecology", "Orthopaedics", "General Surgery", "ENT"]
subscriptions = ["Gold", "Platinum", "Silver", "Bronze"]
upselling_phrases = [
    "You might want to consider", "A better option would be", "I recommend upgrading to",
    "You could save more by", "This plan offers better benefits", "Would you like to try our premium plan?"
]

# Process Audio Files
@st.cache_data
def process_audio_files():
    data = []
    mp3_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".mp3")]

    for file in mp3_files:
        file_path = os.path.join(INPUT_FOLDER, file)
        print(f"Processing {file}...")

        # Get Audio Duration
        audio_length = librosa.get_duration(path=file_path)

        # Transcription
        result = whisper_model.transcribe(file_path)
        text = result["text"]

        # Extract Features
        subscription, medical_test, emergency, problem, report_delay, upselling = "", "", "No", "No", "No", "No"

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

        data.append([file, round(audio_length, 2), subscription, medical_test, problem, emergency, report_delay, upselling, sentiment])

    df = pd.DataFrame(data, columns=["File", "Audio Length (sec)", "Subscription", "Medical Test", "Problem", "Emergency", "Report Delay", "Upselling", "Sentiment"])
    df.to_csv(OUTPUT_CSV, index=False)
    return df

# Load Data
if "input_folder" in st.session_state:
    df = process_audio_files()
    st.write("## Processed Data")
    st.dataframe(df.head())

    # Sentiment Analysis
    st.write("## Sentiment Analysis")
    sentiment_counts = df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    # Upselling Analysis
    st.write("## Upselling Analysis")
    upselling_counts = df["Upselling"].value_counts()
    st.bar_chart(upselling_counts)

    # Long Calls
    st.write("## Long Calls")
    st.dataframe(df[df["Audio Length (sec)"] > 500])

    st.success("✅ Analysis Complete!")
else:
    st.warning("⚠️ Please connect to FTP and download files before processing.")
