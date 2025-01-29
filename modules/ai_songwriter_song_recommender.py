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

def compute_sentence_vector(lyrics, model):
    """
    Compute the sentence vector by averaging word vectors.
    """
    words = lyrics.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]

    if word_vectors:
        return sum(word_vectors) / len(word_vectors)
    else:
        return [0] * model.vector_size

@st.cache_data
def load_sample_dataframe():
    """
    Load a sample DataFrame with song data, including title, artist, lyrics, and mood.
    """
    connection_string = "mysql+mysqlconnector://ironhack:{}@127.0.0.1/spotify_lyrics_db".format(SPOTIFY_DB)
    engine = create_engine(connection_string)

    query = "SELECT song_name, artist, spotify_link, mood, lyrics, mood_strength, popularity FROM dim_tracks_details;"
    df = pd.read_sql(query, con=engine)

    return df

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
        error = f"❌ Error occurred: {e}\nTraceback: {tb.format_exc()}"
        st.error(error)
        return pd.DataFrame()

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

def generate_song_lyrics(selected_artist, selected_song, selected_lyrics, prompt, mood):
    """
    Generate song lyrics based on the user-selected artist, song, mood, and custom prompt.
    """
    try:
        system_prompt = f"""
        You are a creative AI assistant skilled at writing song lyrics. 
        Based on the user's input and the selected inspiration, generate a new song. 
        Ensure the song aligns with the following details:
        - Mood: {mood}
        - Inspiration artist: {selected_artist}
        - Inspiration song: {selected_song}
        - Inspiration lyrics: "{selected_lyrics}"
        - Prompt: "{prompt}"
        The song should include a song name, verses, a chorus, and, optionally, a bridge.
        Write the similarity score of how much you were inspired from the original "{selected_song}" and mention the song name and the artist of inspiration.
        Strictly Maintain the following format for the response, No need to mention the details of similarity:
        Song Name
        Lyrics
        Inspiration
        Similarity Score
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a song inspired by {selected_song} by {selected_artist} with the mood '{mood}' and the following lyrics:\n\n{selected_lyrics}"}
        ]

        completion = openai.OpenAI().chat.completions.create(messages=messages, model='gpt-4', temperature=0.7, max_tokens=2000)

        lyrics = completion.choices[0].message.content
        return lyrics

    except Exception as e:
        error = f"❌ Error occurred: {e}\nTraceback: {tb.format_exc()}"
        st.error(error)
        return "An error occurred while generating the song lyrics."

def predict_song_popularity(lyrics):
    """
    Predict the popularity of a song based on its lyrics using the random forest model and FastText embeddings.
    """
    try:
        # Compute the sentence vector for the lyrics
        lyrics_embedding = compute_sentence_vector(lyrics, ft_model)

        # Make prediction using the random forest model
        popularity_score = rf_model.predict([lyrics_embedding])[0]

        return popularity_score

    except Exception as e:
        error = f"❌ Error occurred: {e}\nTraceback: {tb.format_exc()}"
        st.error(error)
        return "An error occurred while predicting song popularity."