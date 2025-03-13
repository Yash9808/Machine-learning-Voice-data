import os
import ftplib
import assemblyai as aai
import streamlit as st
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import whisper
from ftplib import FTP

# FTP host
host = "cph.v4one.co.uk"

# Transcription Function
def transcribe_audio(file_path, api_key):
    try:
        aai.settings.api_key = api_key
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path)
        return transcript.text
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

# FTP Connection Function
def connect_ftp(ftp_user, ftp_pass):
    try:
        ftp = ftplib.FTP(host)
        ftp.login(ftp_user, ftp_pass)
        folders = ftp.nlst()
        ftp.quit()
        return folders, None
    except Exception as e:
        return None, str(e)

# Fetch Audio Files from FTP
def fetch_from_ftp(ftp_user, ftp_pass, ftp_folder, api_key):
    try:
        ftp = ftplib.FTP(host)
        ftp.login(ftp_user, ftp_pass)
        ftp.cwd(ftp_folder)
        filenames = ftp.nlst()
        results = {}
        
        os.makedirs("./downloads", exist_ok=True)
        
        for filename in filenames:
            local_path = f"./downloads/{filename}"
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f'RETR {filename}', f.write)
            st.write(f"Downloaded file: {filename}")
            transcription = transcribe_audio(local_path, api_key)
            if transcription:
                results[filename] = transcription
            else:
                st.write(f"Failed to transcribe {filename}")
        
        ftp.quit()
        return results
    except Exception as e:
        st.error(f"Error during FTP fetch: {e}")
        return {}

# Main Streamlit Application
st.title("FTP & Manual Audio Transcriber with AssemblyAI")

api_key = st.text_input("Enter your AssemblyAI API Key", type="password")

st.subheader("Login via FTP or Upload Files Manually")

ftp_login_success = False
ftp_user = st.text_input("FTP Username")
ftp_pass = st.text_input("FTP Password", type="password")

# FTP Connection and Fetching Files
if st.button("Connect to FTP"):
    folders, error = connect_ftp(ftp_user, ftp_pass)
    if error:
        st.error(f"Failed to connect: {error}")
    else:
        ftp_login_success = True
        st.success("Connected to FTP Server!")
        folder = st.selectbox("Select an FTP folder", folders)
        if st.button("Fetch & Transcribe Audio Files"):
            if not api_key:
                st.error("Please enter your AssemblyAI API Key")
            else:
                with st.spinner("Fetching and transcribing files..."):
                    results = fetch_from_ftp(ftp_user, ftp_pass, folder, api_key)
                    if results:
                        for filename, transcript in results.items():
                            st.subheader(f"Transcription for {filename}")
                            # Add unique key for each text_area to avoid the duplicate ID error
                            st.text_area(f"Transcript for {filename}", transcript, height=200, key=filename)
                    else:
                        st.error("No transcriptions found or error occurred.")

# Uploading and Transcribing Audio Files Manually
st.subheader("Or Upload Audio Files Manually")
uploaded_files = st.file_uploader("Upload audio files", accept_multiple_files=True)

if st.button("Transcribe Uploaded Files"):
    if not api_key:
        st.error("Please enter your AssemblyAI API Key")
    elif not uploaded_files:
        st.error("Please upload at least one file")
    else:
        results = {}
        os.makedirs("./uploads", exist_ok=True)
        for file in uploaded_files:
            file_path = os.path.join("./uploads", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            results[file.name] = transcribe_audio(file_path, api_key)
        
        # Display the transcriptions with unique keys for each text_area
        for filename, transcript in results.items():
            st.subheader(f"Transcription for {filename}")
            st.text_area(f"Transcript for {filename}", transcript, height=200, key=filename)

# Function to process the audio files and generate transition matrix and plots
@st.cache_data
def process_audio_files():
    INPUT_FOLDER = "audio_files"
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
        result = whisper_model.transcribe(file_path)
        text = result.get("text", "").strip()
        data.append([file, text])

        # Delete the MP3 file after processing to save space
        os.remove(file_path)
        st.write(f"Deleted file: {file}")

    df = pd.DataFrame(data, columns=["File", "Transcription"])
    
    # Example columns for features like Subscription, Medical Test, etc.
    df['Subscription'] = np.random.choice([0, 1], size=len(df))  # Replace with actual data
    df['Medical Test'] = np.random.choice([0, 1], size=len(df))
    df['Emergency'] = np.random.choice([0, 1], size=len(df))
    df['Problem'] = np.random.choice([0, 1], size=len(df))
    df['Report Delay'] = np.random.choice([0, 1], size=len(df))
    df['Upselling'] = np.random.choice([0, 1], size=len(df))
    df['Sentiment'] = np.random.choice([0, 1], size=len(df))
    df['Issue'] = np.random.choice([0, 1], size=len(df))

    # Create 'State' column as a tuple of the feature values
    df['State'] = df[['Subscription', 'Medical Test', 'Emergency', 'Problem', 'Report Delay', 'Upselling', 'Sentiment', 'Issue']].apply(tuple, axis=1)

    # Filter for long calls
    df['Long Call'] = df['Audio Length (sec)'] > 500  # Assuming 'Audio Length (sec)' exists in your data

    # Generate transition matrix (Markov Chain)
    transition_matrix = defaultdict(lambda: defaultdict(int))
    sales_increase_matrix = defaultdict(lambda: defaultdict(int))  # Track sales increase occurrences

    # Assuming 'Sales Increase' column exists
    df['Sales Increase'] = np.random.choice([0, 1], size=len(df))  # Example, replace with actual column if available

    # Create transition matrix
    for i in range(len(df) - 1):
        current_state = df.iloc[i]['State']
        next_state = df.iloc[i + 1]['State']
        sales_increase = df.iloc[i + 1]['Sales Increase']

        transition_matrix[current_state][next_state] += 1
        if sales_increase:
            sales_increase_matrix[current_state][next_state] += 1

    # Normalize transition probabilities
    for current_state, next_states in transition_matrix.items():
        total_count = sum(next_states.values())
        for next_state, count in next_states.items():
            transition_matrix[current_state][next_state] = count / total_count

    # Collect data for plotting
    states = list(transition_matrix.keys())
    all_next_states = set(state for next_states in transition_matrix.values() for state in next_states)

    transition_data = []
    sales_data = []

    for current_state in states:
        for next_state in all_next_states:
            transition_data.append(transition_matrix[current_state].get(next_state, 0))
            sales_data.append(sales_increase_matrix[current_state].get(next_state, 0))

    # Create DataFrame for transition data
    transition_df = pd.DataFrame({
        'State': [str(state) for state in states for _ in all_next_states],
        'Next State': [str(next_state) for _ in states for next_state in all_next_states],
        'Transition Probability': transition_data,
        'Sales Increase Occurrence': sales_data
    })

    # Generate heatmap for transition probabilities
    plt.figure(figsize=(12, 8))
    sns.heatmap(transition_df.pivot(index='State', columns='Next State', values='Transition Probability'),
                annot=True, fmt='.2f', cmap="YlGnBu", cbar_kws={'label': 'Transition Probability'})

    plt.title('Transition Probabilities with Sales Increase Data', fontsize=16)
    plt.xlabel('Next State', fontsize=12)
    plt.ylabel('Current State', fontsize=12)
    plt.show()

    # Optional: Plot Sales Increase Occurrence as a separate heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(transition_df.pivot(index='State', columns='Next State', values='Sales Increase Occurrence'),
                annot=True, fmt='.0f', cmap="Reds", cbar_kws={'label': 'Sales Increase Occurrence'})

    plt.title('Sales Increase Occurrences in State Transitions', fontsize=16)
    plt.xlabel('Next State', fontsize=12)
    plt.ylabel('Current State', fontsize=12)
    plt.show()

    # Scatter plot for 'Audio Length (sec)' vs State
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df.index, y='Audio Length (sec)', data=df, hue='State', palette="Set1", legend=None)
    long_call_df = df[df['Long Call']]
    plt.scatter(long_call_df.index, long_call_df['Audio Length (sec)'], color='red', s=100, label='Long Call (>500 sec)', edgecolor='black', marker='o')

    issue_state_df = df[df['Issue'] == 1]  # Assuming Issue is 1 for problem calls
    plt.scatter(issue_state_df.index, issue_state_df['Audio Length (sec)'], color='none', s=200, label='Issue Call', edgecolor='black', marker='o')

    plt.title('Audio Length vs State with Long Calls and Issue Calls Marked', fontsize=16)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Audio Length (sec)', fontsize=12)
    plt.legend()
    plt.show()

    return df

# Set input folder and process the files
whisper_model = whisper.load_model("base")
OUTPUT_CSV = "telecall_analysis.csv"

if "input_folder" in st.session_state:
    df = process_audio_files()
    st.dataframe(df.head())
