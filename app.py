import os
import pandas as pd
import numpy as np
import librosa
import wave
import seaborn as sns
import matplotlib.pyplot as plt
import whisper
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st
from pydub import AudioSegment

# Function to convert MP3 to WAV
def convert_audio_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio = AudioSegment.from_file(mp3_path, format="mp3")
    audio.export(wav_path, format="wav")
    return wav_path

# Load Whisper model
model = whisper.load_model("base")

def transcribe_with_whisper(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # Sentiment score (-1 to +1)
    sentiment = "P" if sentiment_score > 0 else "N" if sentiment_score < 0 else "NU"
    words = text.split()
    positive_words = sum(1 for word in words if TextBlob(word).sentiment.polarity > 0)
    negative_words = sum(1 for word in words if TextBlob(word).sentiment.polarity < 0)
    return sentiment, sentiment_score, positive_words, negative_words

# Function to analyze voice tone
def analyze_tone(audio_file):
    y, sr = librosa.load(audio_file)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    energy = np.mean(librosa.feature.rms(y=y))

    if avg_pitch > 200 and energy > 0.1:
        return "Happy", 1
    elif avg_pitch > 150 and energy > 0.08:
        return "Calm", 0.5
    elif avg_pitch < 100 and energy > 0.1:
        return "Angry", -1
    elif avg_pitch < 100 and energy < 0.05:
        return "Sad", -0.5
    else:
        return "Frustrated", -0.8

# Streamlit app interface
st.title("ML Outcome of Bulk Data")
uploaded_files = st.file_uploader("Choose Audio Files", type=["mp3"], accept_multiple_files=True)

if uploaded_files:
    sentiment_results = []
    tone_results = []
    
    for uploaded_file in uploaded_files:
        st.write(f"Processing: {uploaded_file.name}")
        
        audio_path = os.path.abspath(f"temp_{uploaded_file.name}")
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        audio_path_wav = convert_audio_to_wav(audio_path)
        text = transcribe_with_whisper(audio_path_wav)
        agent_sentiment, agent_score, apw, anw = analyze_sentiment(text)
        agent_tone, agent_tone_score = analyze_tone(audio_path_wav)
        
        sentiment_results.append([uploaded_file.name, agent_sentiment, agent_score, apw, anw])
        tone_results.append([uploaded_file.name, agent_tone, agent_tone_score])
        
        os.remove(audio_path)
        os.remove(audio_path_wav)
    
    df_sentiment = pd.DataFrame(sentiment_results, columns=["File", "Sentiment", "Score", "Pos Words", "Neg Words"])
    df_tone = pd.DataFrame(tone_results, columns=["File", "Tone", "Tone Score"])
    
    df_sentiment.to_csv("sentiment_results.csv", index=False)
    df_tone.to_csv("tone_results.csv", index=False)
    
    st.write("Sentiment and tone analysis complete!")
    
    # Display results
    st.dataframe(df_sentiment)
    st.dataframe(df_tone)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    sns.histplot(df_sentiment['Score'], bins=20, kde=True, color='blue')
    plt.title("Sentiment Score Distribution")
    st.pyplot()
    
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Tone', data=df_tone, palette='coolwarm')
    plt.title("Tone Classification")
    st.pyplot()
