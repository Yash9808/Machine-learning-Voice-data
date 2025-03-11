import os
import pandas as pd
import numpy as np
import librosa
import whisper
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from textblob import TextBlob
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Load AI models
whisper_model = whisper.load_model("base")

# Define paths
INPUT_FOLDER = "audio_files"  # Change this to your folder
OUTPUT_CSV = "telecall_analysis_130.csv"

# Define keywords
medical_tests = ["MRI", "X-Ray", "Ultrasound", "Endoscopy", "Gynaecology", "Orthopaedics", "General Surgery", "ENT"]
subscriptions = ["Gold", "Platinum", "Silver", "Bronze"]
upselling_phrases = [
    "You might want to consider", "A better option would be", "I recommend upgrading to",
    "You could save more by", "This plan offers better benefits", "Would you like to try our premium plan?"
]

# Process audio files
@st.cache_data
def process_audio_files():
    data = []
    mp3_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".mp3")][:135]
    
    for file in mp3_files:
        file_path = os.path.join(INPUT_FOLDER, file)
        print(f"Processing {file}...")

        # Audio duration
        audio_length = librosa.get_duration(path=file_path)

        # Transcription
        result = whisper_model.transcribe(file_path)
        text = result["text"]

        # Named Entity Recognition (NER) using NLTK
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        entities = [word for word, tag in tagged_words if tag in ["NNP", "NN"]]

        # Extract features
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

        # Sentiment Analysis using TextBlob
        sentiment_score = TextBlob(text).sentiment.polarity
        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

        # Upselling detection
        if any(phrase.lower() in text.lower() for phrase in upselling_phrases):
            upselling = "Yes" if sentiment in ["Positive", "Neutral"] else "No"

        data.append([file, round(audio_length, 2), agent_name, person_name, subscription, medical_test,
                     age, problem, emergency, problem, report_delay, upselling, entities, sentiment])

    df = pd.DataFrame(data, columns=["File", "Audio Length (sec)", "Agent Name", "Person Name", "Subscription",
                                     "Medical Test", "Age", "Issue", "Emergency", "Problem", "Report Delay",
                                     "Upselling", "NER", "Sentiment"])
    df.to_csv(OUTPUT_CSV, index=False)
    return df

# Load and analyze data
def analyze_data(df):
    categorical_cols = ["Subscription", "Medical Test", "Emergency", "Problem", "Report Delay", "Upselling", "Sentiment", "Issue"]
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes
    
    df['State'] = df[categorical_cols].apply(tuple, axis=1)
    df['Long Call'] = df['Audio Length (sec)'] > 500
    transition_matrix = defaultdict(lambda: defaultdict(int))

    for i in range(len(df) - 1):
        current_state = df.iloc[i]['State']
        next_state = df.iloc[i + 1]['State']
        transition_matrix[current_state][next_state] += 1

    states = list(transition_matrix.keys())
    all_next_states = set(state for next_states in transition_matrix.values() for state in next_states)

    transition_data = []
    for current_state in states:
        for next_state in all_next_states:
            transition_data.append(transition_matrix[current_state].get(next_state, 0))

    transition_df = pd.DataFrame({
        'State': [str(state) for state in states for _ in all_next_states],
        'Next State': [str(next_state) for _ in states for next_state in all_next_states],
        'Transition Probability': transition_data
    })
    return df, transition_df

# Streamlit App
st.title("Telecall Analysis Dashboard")

if not os.path.exists(OUTPUT_CSV):
    st.warning("Processing audio files, this might take a while...")
    df = process_audio_files()
else:
    df = pd.read_csv(OUTPUT_CSV)

df, transition_df = analyze_data(df)

st.write("## Processed Data")
st.dataframe(df.head())

st.write("## Transition Matrix Heatmap")
plt.figure(figsize=(12, 8))
sns.heatmap(transition_df.pivot(index='State', columns='Next State', values='Transition Probability'),
            annot=True, fmt='.2f', cmap="YlGnBu")
st.pyplot(plt)

st.write("## Audio Length Distribution")
plt.figure(figsize=(10, 5))
sns.histplot(df['Audio Length (sec)'], bins=20, kde=True)
st.pyplot(plt)

st.write("## Sentiment Analysis")
sentiment_counts = df['Sentiment'].value_counts()
st.bar_chart(sentiment_counts)

st.write("## Upselling Analysis")
upselling_counts = df['Upselling'].value_counts()
st.bar_chart(upselling_counts)

st.write("## Long Calls")
st.dataframe(df[df['Long Call']])

st.success("Analysis Complete!")
