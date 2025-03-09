import streamlit as st
import os
import pandas as pd
import librosa
import speech_recognition as sr
import whisper
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import pipeline  # Hugging Face's pipeline for NER

# Initialize Hugging Face NER pipeline
ner_pipeline = pipeline("ner")

# Load Whisper model for transcription
model = whisper.load_model("base")

# Streamlit UI
st.title("Call Analysis & Prediction App")
st.sidebar.header("Upload Call Recording or CSV Data")
uploaded_file = st.sidebar.file_uploader("Upload an MP3 file", type=["mp3"])
uploaded_csv = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    file_path = f"temp.mp3"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("File uploaded successfully!")

    # Transcription
    st.subheader("Transcription")
    result = model.transcribe(file_path)
    transcription = result["text"]
    st.write(transcription)
    
    # Named Entity Recognition (NER) using Hugging Face transformers
    entities = ner_pipeline(transcription)
    st.subheader("Named Entities")
    st.write(entities)
    
    # Sentiment Analysis
    sentiment = TextBlob(transcription).sentiment.polarity
    st.subheader("Sentiment Score")
    st.write(sentiment)
    
    # Tone Analysis (Pitch & Intensity)
    y, sr = librosa.load(file_path)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    avg_pitch = np.nanmean(pitch)
    st.subheader("Tone Analysis")
    st.write(f"Average Pitch: {avg_pitch:.2f} Hz")
    
    # Save Data
    df = pd.DataFrame([{"File": uploaded_file.name, "Transcription": transcription, "Sentiment": sentiment, "Avg Pitch": avg_pitch}])
    df.to_csv("call_analysis.csv", index=False)
    
    # Visualization
    st.subheader("Data Visualization")
    sns.histplot(df["Sentiment"], kde=True)
    st.pyplot()

if uploaded_csv:
    st.subheader("Uploaded CSV Data")
    data = pd.read_csv(uploaded_csv)
    st.write(data.head())

    # Prepare Data for ML
    X = data.drop(columns=["File", "Agent Name", "Customer Name", "Overall Sentiment"])
    y = data["Overall Sentiment"].apply(lambda x: 1 if x == "positive" else 0)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    # Machine Learning Model (LSTM)
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])
    
    # Train LSTM on Real Data
    model = LSTMModel(X.shape[1], 10, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(5):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
    
    st.subheader("Machine Learning Model Trained")
    st.write("LSTM Model has been trained on uploaded CSV data.")
