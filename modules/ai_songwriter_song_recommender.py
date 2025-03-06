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
from modules import spotify_lyrics_dataset_generator_helper as sld
import joblib
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

SPOTIFY_DB = os.getenv("SPOTIFY_DB_PASS")

# Load pre-trained model and FastText embeddings
@st.cache_resource
def load_model_and_embeddings():
    model_files_path = './model_files'
    model_path = os.path.join(model_files_path, "rf_model.pkl") 
    embeddings_path = os.path.join(model_files_path, "fasttext_model.bin") 

    rf_model = joblib.load(model_path)
    ft_model = FastText.load(embeddings_path)
    
    return rf_model, ft_model

rf_model, ft_model = load_model_and_embeddings()

# Compute sentence vector for lyrics
def compute_sentence_vector(lyrics, model):
    words = lyrics.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]

    if word_vectors:
        return sum(word_vectors) / len(word_vectors)
    else:
        return [0] * model.vector_size


# Load embedding model for RAG
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

@st.cache_data
def load_lyrics_dataframe():
    connection_string = "mysql+mysqlconnector://ironhack:{}@127.0.0.1/spotify_lyrics_db".format(SPOTIFY_DB)
    engine = create_engine(connection_string)
    query = "SELECT song_name, artist, lyrics, mood, mood_strength, popularity, spotify_link FROM dim_tracks_details;"
    df = pd.read_sql(query, con=engine)
    return df

lyrics_df = load_lyrics_dataframe()

def recommend_songs_by_mood(selected_mood, df, num_songs):
    """
    Recommend songs based on the selected mood. Select a random subset, then rank by mood_strength.
    """
    try:
        filtered_songs = df[df['mood'] == selected_mood.lower()]
        filtered_songs = filtered_songs.drop_duplicates(subset=['song_name', 'artist'], keep='first')
        shuffled_songs = filtered_songs.sample(num_songs)
        recommendations = shuffled_songs.sort_values(by=['mood_strength','popularity'], ascending=[False,False])
        return recommendations

    except Exception as e:
        st.error(f"❌ Error occurred while recommending songs: {e}")
        return pd.DataFrame()

# Initialize ChromaDB client
@st.cache_resource
def init_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="lyrics_embeddings")
    return collection

chroma_collection = init_chroma()

@st.cache_data
def get_retrieved_lyrics(user_input, top_k=5):
    """Retrieve similar lyrics using ChromaDB."""
    try:
        user_embedding = embedding_model.encode([user_input]).tolist()
        results = chroma_collection.query(query_embeddings=user_embedding, n_results=top_k)
        retrieved_lyrics = [res["lyrics"] for res in results["metadatas"][0]]
        return retrieved_lyrics
    except Exception as e:
        st.error(f"Error retrieving lyrics: {e}")
        return []

def map_prompt_to_mood(prompt):
    """
    Map user input prompt to a mood dynamically based on keywords.
    """
    mood_mapping = {
                        "Happy": [
                            "joy", "celebrate", "happiness", "cheerful", "smile", "laughter", 
                            "delight", "bright", "euphoria", "sunshine", "grateful", "content", "bliss"
                        ],
                        "Melancholy": [
                            "lost", "lonely", "cry", "heartbroken", "tears", "pain", "sorrow", 
                            "despair", "wistful", "aching", "regret", "longing", "dreary", "fading"
                        ],
                        "Energetic": [
                            "dance", "party", "excited", "upbeat", "move", "rush", "fast", "power", 
                            "drive", "hype", "thrill", "jump", "burst", "electric", "frenzy"
                        ],
                        "Calm": [
                            "peaceful", "relaxed", "quiet", "serenity", "soft", "breeze", "gentle", 
                            "tranquil", "stillness", "meditative", "harmony", "soothing", "ease"
                        ],
                        "Romantic": [
                            "love", "romance", "affection", "heart", "kiss", "embrace", "passion", 
                            "desire", "intimacy", "tender", "forever", "flame", "devotion", "enchanted"
                        ],
                        "Hopeful": [
                            "dream", "rise", "aspire", "promise", "brighter", "believe", "uplift", 
                            "shining", "new", "renew", "dawn", "possibility", "soar", "courage"
                        ],
                        "Nostalgic": [
                            "memory", "past", "golden", "old", "bittersweet", "reminisce", 
                            "days gone", "echo", "sepia", "fading", "yearning", "cherish", "horizon"
                        ],
                        "Dark": [
                            "shadow", "fear", "cold", "void", "mystery", "haunted", "nightmare", 
                            "whisper", "alone", "ghostly", "doom", "abyss", "eerie", "cursed"
                        ],
                        "Euphoric": [
                            "ecstasy", "high", "limitless", "elevate", "transcend", "glow", 
                            "elation", "dizzy", "rush", "rapture", "euphoria", "elevated", "surge"
                        ]
    }

    prompt_lower = prompt.lower()

    for mood, keywords in mood_mapping.items():
        if any(keyword in prompt_lower for keyword in keywords):
            return mood

    return "Neutral"

def generate_song_lyrics(selected_artist, selected_song, selected_lyrics, prompt, mood, use_rag=True):
    """
    Generate song lyrics with or without retrieval-augmented generation (RAG).
    """
    try:
        
        retrieved_lyrics_text = get_retrieved_lyrics(prompt) if use_rag else []
        
        system_prompt = f"""
                You are a creative AI assistant skilled at writing song lyrics. 
                Based on the user's input {"and retrieved similar lyrics" if use_rag else ""}, generate a new song.
                Ensure the song aligns with:
                - Mood: {mood}
                - Inspiration artist: {selected_artist}
                - Inspiration song: {selected_song}
                - User input: "{prompt}"
                {retrieved_lyrics_text}
                The song should include a title, verses, a chorus, and optionally a bridge.
                Strictly Maintain the following format:
                Song Name
                Lyrics
                Inspiration
                """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a song inspired by {selected_song} by {selected_artist} with the mood '{mood}' and the following lyrics:\n\n {retrieved_lyrics_text.append(selected_lyrics) if use_rag else selected_lyrics}"}
        ]
        
        completion = openai.OpenAI().chat.completions.create(messages=messages, model='gpt-4', temperature=0.7, max_tokens=2000)

        return completion.choices[0].message.content

    except Exception as e:
        error = f"❌ Error occurred: {e}\nTraceback: {tb.format_exc()}"
        st.error(error)
        return "An error occurred while generating the song lyrics."

def predict_song_popularity(lyrics):
    """
    Predict the popularity of a song based on its lyrics using the random forest model and FastText embeddings.
    """
    try:
        lyrics_embedding = compute_sentence_vector(lyrics, ft_model)
        popularity_score = rf_model.predict([lyrics_embedding])[0]
        return popularity_score
    except Exception as e:
        error = f"❌ Error occurred: {e}\nTraceback: {tb.format_exc()}"
        st.error(error)
        return "An error occurred while predicting song popularity."
