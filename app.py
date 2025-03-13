import os
import ftplib
import assemblyai as aai
import streamlit as st
import librosa
import soundfile as sf
import pandas as pd
import whisper
from ftplib import FTP
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from textblob import TextBlob

host = "cph.v4one.co.uk"

def transcribe_audio(file_path, api_key):
    aai.settings.api_key = api_key
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    return transcript.text

def connect_ftp(ftp_user, ftp_pass):
    try:
        ftp = ftplib.FTP(host)
        ftp.login(ftp_user, ftp_pass)
        folders = ftp.nlst()
        ftp.quit()
        return folders, None
    except Exception as e:
        return None, str(e)

def fetch_from_ftp(ftp_user, ftp_pass, ftp_folder, api_key):
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
        results[filename] = transcribe_audio(local_path, api_key)
    
    ftp.quit()
    return results

st.title("FTP & Manual Audio Transcriber with AssemblyAI")

api_key = st.text_input("Enter your AssemblyAI API Key", type="password")

st.subheader("Login via FTP or Upload Files Manually")

ftp_login_success = False
ftp_user = st.text_input("FTP Username")
ftp_pass = st.text_input("FTP Password", type="password")

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
                    for filename, transcript in results.items():
                        st.subheader(f"Transcription for {filename}")
                        st.text_area("", transcript, height=200)

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
        
        for filename, transcript in results.items():
            st.subheader(f"Transcription for {filename}")
            st.text_area("Transcript", transcript, height=200)

st.subheader("Select Date & Download Files")
if "available_dates" in st.session_state:
    selected_date = st.selectbox("📅 Select a Date", st.session_state["available_dates"])
    
    if st.button("📥 Download & Process Audio"):
        try:
            ftp = FTP(host)
            ftp.login(user=ftp_user, passwd=ftp_pass)
            remote_folder = selected_date
            local_folder = f"audio_files/{selected_date}"
            os.makedirs(local_folder, exist_ok=True)
            audio_files = []
            ftp.cwd(remote_folder)
            ftp.retrlines("LIST", lambda x: audio_files.append(x.split()[-1]))
            for file in audio_files:
                local_file_path = os.path.join(local_folder, file)
                with open(local_file_path, "wb") as f:
                    ftp.retrbinary(f"RETR {file}", f.write)
            ftp.quit()
            st.success(f"✅ Downloaded {len(audio_files)} files from {selected_date}")
            st.session_state["input_folder"] = local_folder
        except Exception as e:
            st.error(f"Download failed: {e}")

INPUT_FOLDER = st.session_state.get("input_folder", "audio_files")
whisper_model = whisper.load_model("base")
OUTPUT_CSV = "telecall_analysis.csv"

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
        result = whisper_model.transcribe(file_path)
        text = result.get("text", "").strip()
        data.append([file, text])
    df = pd.DataFrame(data, columns=["File", "Transcription"])
    df.to_csv(OUTPUT_CSV, index=False)
    return df

if "input_folder" in st.session_state:
    df = process_audio_files()
    st.dataframe(df.head())

if __name__ == "__main__":
    st.title(" ")
