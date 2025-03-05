import streamlit as st
import pandas as pd
import os
import openai
import pickle
from gensim.models import FastText
from dotenv import load_dotenv
from sqlalchemy import create_engine
import traceback as tb
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import joblib
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
SPOTIFY_DB = os.getenv("SPOTIFY_DB_PASS")

# Load pre-trained models
@st.cache_resource
def load_model_and_embeddings():
    model_files_path = './model_files'
    model_path = os.path.join(model_files_path, "rf_model.pkl") 
    embeddings_path = os.path.join(model_files_path, "fasttext_model.bin") 

    rf_model = joblib.load(model_path)
    ft_model = FastText.load(embeddings_path)
    return rf_model, ft_model

rf_model, ft_model = load_model_and_embeddings()

# Load embedding model for RAG
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

@st.cache_data
def load_lyrics_dataframe():
    connection_string = "mysql+mysqlconnector://ironhack:{}@127.0.0.1/spotify_lyrics_db".format(SPOTIFY_DB)
    engine = create_engine(connection_string)
    query = "SELECT song_name, artist, lyrics FROM dim_tracks_details;"
    df = pd.read_sql(query, con=engine)
    return df

lyrics_df = load_lyrics_dataframe()

# Initialize ChromaDB client
@st.cache_resource
def init_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="lyrics_embeddings")
    return collection

chroma_collection = init_chroma()

# Store lyrics embeddings in ChromaDB
@st.cache_resource
def store_lyrics_in_chroma():
    existing_items = chroma_collection.count()
    if existing_items == 0:
        for i, row in lyrics_df.iterrows():
            embedding = embedding_model.encode(row['lyrics']).tolist()
            chroma_collection.add(ids=[str(i)], embeddings=[embedding], metadatas=[{"song_name": row['song_name'], "artist": row['artist'], "lyrics": row['lyrics']}])

store_lyrics_in_chroma()

@st.cache_data
def get_retrieved_lyrics(user_input, top_k=3):
    """Retrieve similar lyrics using ChromaDB."""
    try:
        user_embedding = embedding_model.encode([user_input]).tolist()
        results = chroma_collection.query(query_embeddings=user_embedding, n_results=top_k)
        retrieved_lyrics = [res["lyrics"] for res in results["metadatas"][0]]
        return retrieved_lyrics
    except Exception as e:
        st.error(f"Error retrieving lyrics: {e}")
        return []

def format_lyrics(lyrics):
    """Formats the generated lyrics in a professional and visually appealing style."""
    formatted = lyrics.replace("\n", "<br>")
    return f"""
    <div style='background-color: #181818; padding: 20px; border-radius: 10px;'>
        <h2 style='color: #1DB954; text-align: center; font-family: Arial, sans-serif;'>🎵 Generated Song 🎵</h2>
        <p style='font-family: Arial, sans-serif; color: #FFFFFF; font-size: 18px; line-height: 1.8;'>{formatted}</p>
    </div>
    """

def generate_song_lyrics(selected_artist, selected_song, selected_lyrics, prompt, mood):
    try:
        retrieved_lyrics = get_retrieved_lyrics(prompt)
        system_prompt = f"""
        You are a creative AI assistant skilled at writing song lyrics. 
        Based on the user's input and retrieved similar lyrics, generate a new song.
        Ensure the song aligns with:
        - Mood: {mood}
        - Inspiration artist: {selected_artist}
        - Inspiration song: {selected_song}
        - User input: "{prompt}"
        - Retrieved lyrics for inspiration: "{retrieved_lyrics}"
        The song should include a title, verses, a chorus, and optionally a bridge.
        Strictly Maintain the following format:
        Song Name
        Lyrics
        Inspiration
        Similarity Score
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a song inspired by {selected_song} by {selected_artist} with the mood '{mood}' and the following lyrics:\n\n{retrieved_lyrics}"}
        ]
        
        completion = openai.OpenAI().chat.completions.create(messages=messages, model='gpt-4', temperature=0.7, max_tokens=2000)

        return completion.choices[0].message.content

    except Exception as e:
        error = f"❌ Error occurred: {e}\nTraceback: {tb.format_exc()}"
        st.error(error)
        return "An error occurred while generating the song lyrics."

# UI Layout
st.title("🎼 AI Songwriter with RAG 🎤")

# Dropdowns for user selection
selected_artist = st.selectbox("🎶 Select an artist:", lyrics_df['artist'].unique())
filtered_songs = lyrics_df[lyrics_df['artist'] == selected_artist]
selected_song = st.selectbox("🎵 Select a song:", filtered_songs['song_name'].unique())
selected_lyrics = filtered_songs[filtered_songs['song_name'] == selected_song]['lyrics'].values[0]

user_input = st.text_area("✍️ Enter a theme or lyrics snippet for inspiration:")
mood = st.selectbox("🎭 Select Mood:", ["Happy", "Melancholy", "Energetic", "Calm", "Romantic", "Hopeful", "Nostalgic", "Dark", "Euphoric"])

# Buttons Layout
col1, col2 = st.columns([4, 1])
with col1:
    if st.button("🎼 Generate Song"):
        retrieved_lyrics = get_retrieved_lyrics(user_input)
        st.write("**🔍 Retrieved similar lyrics for context:**")
        for lyric in retrieved_lyrics:
            st.write(f"- {lyric}")
        
        generated_lyrics = generate_song_lyrics(selected_artist, selected_song, selected_lyrics, user_input, mood)
        
        # Display formatted lyrics
        st.markdown(format_lyrics(generated_lyrics), unsafe_allow_html=True)

        # Provide a properly formatted download button
        st.download_button(
            label="📥 Download Lyrics",
            data=generated_lyrics,
            file_name="generated_lyrics.txt",
            mime="text/plain",
            help="Click to download the generated lyrics."
        )

with col2:
    if st.button("🔄 Reset"):
        st.rerun()
