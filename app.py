import os
import pandas as pd
import numpy as np
import librosa
import wave
import seaborn as sns
import matplotlib.pyplot as plt
import speech_recognition as sr
import whisper  # Import Whisper
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

# Function to check if audio file is valid
def check_audio_validity(audio_file):
    try:
        with wave.open(audio_file, 'r') as wav:
            if wav.getnframes() == 0:
                return False  # No audio data
        return True
    except Exception:
        return False

# Function to perform speech-to-text sentiment analysis
def analyze_sentiment(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)  # Convert speech to text
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity  # Sentiment score (-1 to +1)
            sentiment = "P" if sentiment_score > 0 else "N" if sentiment_score < 0 else "NU"

            # Count positive and negative words
            words = text.split()
            positive_words = sum(1 for word in words if TextBlob(word).sentiment.polarity > 0)
            negative_words = sum(1 for word in words if TextBlob(word).sentiment.polarity < 0)

            return text, sentiment, sentiment_score, positive_words, negative_words
        except (sr.UnknownValueError, sr.RequestError):
            return "", "NU", 0, 0, 0  # Unable to transcribe

# Function to analyze voice tone
def analyze_tone(audio_file):
    y, sr = librosa.load(audio_file)

    # Calculate pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    # Calculate intensity (loudness)
    energy = np.mean(librosa.feature.rms(y=y))

    # Classify Tone Based on Pitch & Intensity
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
st.title("ML outcome of Bulk Data")

uploaded_files = st.file_uploader("Choose Audio Files", type=["mp3"], accept_multiple_files=True)

# Load Whisper model for transcription
model = whisper.load_model("base")

def transcribe_with_whisper(audio_file):
    result = model.transcribe(audio_file)
    return result["text"]

if uploaded_files:
    # Data storage
    sentiment_results = []
    tone_results = []

    for uploaded_file in uploaded_files:
        st.write(f"Processing: {uploaded_file.name}")

        # Convert MP3 to text using Whisper (no need to convert to WAV)
        audio_path = f"temp_{uploaded_file.name}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Save the uploaded MP3 file temporarily
        
        text = transcribe_with_whisper(audio_path)  # Transcribe the audio to text using Whisper

        # Perform sentiment analysis
        agent_text, agent_sentiment, agent_sentiment_score, apw, anw = analyze_sentiment(text)
        customer_text, customer_sentiment, customer_sentiment_score, cpw, cnw = analyze_sentiment(text)
        agent_tone, agent_tone_score = analyze_tone(audio_path)
        
        # Save sentiment results
        sentiment_results.append([uploaded_file.name, agent_sentiment, customer_sentiment, agent_sentiment_score, customer_sentiment_score])

        # Save tone results
        tone_results.append([uploaded_file.name, agent_tone, agent_tone_score, apw, anw, cpw, cnw])

        os.remove(audio_path)  # Remove temporary MP3 file

    # Create DataFrames for sentiment and tone analysis
    df_sentiment = pd.DataFrame(sentiment_results, columns=["File", "Agent Sentiment", "Customer Sentiment", "Agent Sentiment Score", "Customer Sentiment Score"])
    df_tone = pd.DataFrame(tone_results, columns=["File", "Agent Tone", "Agent Tone Score", "APW", "ANW", "CPW", "CNW"])

    # Save to CSV files
    sentiment_csv = "sentiment_analysis_results.csv"
    tone_csv = "tone_analysis_results.csv"
    df_sentiment.to_csv(sentiment_csv, index=False)
    df_tone.to_csv(tone_csv, index=False)

    st.write(f"Sentiment results saved to {sentiment_csv}")
    st.write(f"Tone results saved to {tone_csv}")

    # -------------------- TRAINING LSTM MODELS ----------------------
    
    # Combine both sentiment and tone data for ML models
    df_combined = pd.merge(df_sentiment, df_tone, on="File")

    # Prepare data for sentiment-based model
    sentiment_features = ['Agent Sentiment Score', 'Customer Sentiment Score']
    sentiment_target = 'Agent Sentiment Score'  # Example target
    X_sentiment = df_combined[sentiment_features].values
    y_sentiment = df_combined[sentiment_target].values

    # Prepare data for tone-based model
    tone_features = ['Agent Tone Score', 'APW', 'ANW', 'CPW', 'CNW']
    tone_target = 'Agent Tone Score'  # Example target
    X_tone = df_combined[tone_features].values
    y_tone = df_combined[tone_target].values

    # Normalize features
    scaler = MinMaxScaler()
    X_sentiment = scaler.fit_transform(X_sentiment)
    X_tone = scaler.fit_transform(X_tone)

    # Reshape for RNN (samples, timesteps, features)
    X_sentiment = X_sentiment.reshape((X_sentiment.shape[0], 1, X_sentiment.shape[1]))
    X_tone = X_tone.reshape((X_tone.shape[0], 1, X_tone.shape[1]))

    # Split the data for both models
    X_sentiment_train, X_sentiment_test, y_sentiment_train, y_sentiment_test = train_test_split(X_sentiment, y_sentiment, test_size=0.4, random_state=42)
    X_tone_train, X_tone_test, y_tone_train, y_tone_test = train_test_split(X_tone, y_tone, test_size=0.4, random_state=42)

    # Build sentiment model (LSTM)
    sentiment_model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X_sentiment_train.shape[1], X_sentiment_train.shape[2])),
        LSTM(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    sentiment_model.compile(optimizer='adam', loss='mse')

    # Train sentiment model
    sentiment_history = sentiment_model.fit(X_sentiment_train, y_sentiment_train, epochs=50, batch_size=16, validation_data=(X_sentiment_test, y_sentiment_test), verbose=1)

    # Build tone model (LSTM)
    tone_model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X_tone_train.shape[1], X_tone_train.shape[2])),
        LSTM(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    tone_model.compile(optimizer='adam', loss='mse')

    # Train tone model
    tone_history = tone_model.fit(X_tone_train, y_tone_train, epochs=50, batch_size=16, validation_data=(X_tone_test, y_tone_test), verbose=1)

    # -------------------- PLOTS ----------------------

    # Plot Sentiment Model Actual vs Predicted
    y_sentiment_pred = sentiment_model.predict(X_sentiment_test)
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(y_sentiment_test)), y_sentiment_test, label='Actual Sentiment Values', marker='o', linestyle='-', color='blue')
    plt.plot(np.arange(len(y_sentiment_pred)), y_sentiment_pred, label='Predicted Sentiment Values', marker='x', linestyle='--', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Model: Actual vs Predicted')
    plt.legend()
    st.pyplot()

    # Plot Tone Model Actual vs Predicted
    y_tone_pred = tone_model.predict(X_tone_test)
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(y_tone_test)), y_tone_test, label='Actual Tone Values', marker='o', linestyle='-', color='blue')
    plt.plot(np.arange(len(y_tone_pred)), y_tone_pred, label='Predicted Tone Values', marker='x', linestyle='--', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Tone Score')
    plt.title('Tone Model: Actual vs Predicted')
    plt.legend()
    st.pyplot()

    # Transition Probabilities Heatmap
    transition_df = pd.DataFrame({
        'State': ['Happy', 'Sad', 'Neutral'],
        'Next State': ['Sad', 'Happy', 'Neutral'],
        'Transition Probability': [0.2, 0.3, 0.5],
        'Emotion Distribution': ['Positive', 'Negative', 'Neutral']
    })
    st.write("Transition Probability Table")
    st.write(transition_df)

