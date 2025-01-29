import requests
import pandas as pd
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect, DetectorFactory
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import contractions
import emoji
import textwrap
from spellchecker import SpellChecker

DetectorFactory.seed = 0  # Ensure consistent language detection results

# Spotify API credentials from environment variables
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Authenticate with Spotify API
client_credentials_manager = SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Cache for lyrics
lyrics_cache = {}

def fn_fetch_lyrics_from_lyrics_ovh(artist, song):
    """
    Fetches lyrics for a given song using the Lyrics.ovh API.

    Args:
        artist (str): Name of the artist.
        song (str): Name of the song.

    Returns:
        str: Lyrics of the song or a message if not found.
    """
    api_url = f"https://api.lyrics.ovh/v1/{artist}/{song}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return data.get("lyrics", "Lyrics not found")
        elif response.status_code == 404:
            return "Lyrics not found"
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error fetching lyrics: {e}"

def fn_is_english(lyrics):
    """
    Determines if the given lyrics are in English using langdetect.

    Args:
        lyrics (str): The lyrics of the song.

    Returns:
        bool: True if the lyrics are in English, False otherwise.
    """
    try:
        language = detect(lyrics)
        return language == "en"
    except Exception:
        return False

def fn_get_lyrics(artist, song):
    """
    Fetches lyrics for a given song using the Lyrics.ovh API and checks if they are in English.

    Args:
        artist (str): Name of the artist.
        song (str): Name of the song.

    Returns:
        str: Lyrics of the song or a message if not found.
    """
    song_key = f"{artist.lower()}_{song.lower()}"
    if song_key in lyrics_cache:
        return lyrics_cache[song_key]

    lyrics = fn_fetch_lyrics_from_lyrics_ovh(artist, song)
    if lyrics != "Lyrics not found" and fn_is_english(lyrics):
        lyrics_cache[song_key] = lyrics
        return lyrics

    lyrics_cache[song_key] = "Non-English or no lyrics"
    return "Non-English or no lyrics"

def fn_fetch_lyrics_concurrently(songs):
    """
    Fetches lyrics concurrently for a list of songs.

    Args:
        songs (list): List of dictionaries containing song and artist names.

    Returns:
        list: List of dictionaries with lyrics added.
    """
    with ThreadPoolExecutor(max_workers=10) as executor:  # Reduced max_workers to prevent rate-limiting
        future_to_song = {
            executor.submit(fn_get_lyrics, song['artist'], song['song_name']): song
            for song in songs
        }
        results = []
        for future in tqdm(as_completed(future_to_song), total=len(future_to_song), desc="Fetching lyrics"):
            song = future_to_song[future]
            try:
                song['lyrics'] = future.result()
            except Exception as e:
                song['lyrics'] = f"Error: {e}"
            if song['lyrics'] != "Non-English or no lyrics":
                results.append(song)
    return results

def fn_fetch_songs_with_lyrics(search_type, search_queries, limit):
    """
    Fetches songs from Spotify based on search type and adds lyrics using the Lyrics.ovh API.

    Args:
        search_type (str): Type of search (e.g., "genre", "year", "artist").
        search_queries (list): List of search queries (e.g., genres, years, or artist names).
        limit (int): Total number of songs to fetch per query.

    Returns:
        DataFrame: A Pandas DataFrame containing song details with lyrics.
    """
    data = []

    for query in tqdm(search_queries, desc="Fetching songs"):
        songs = []
        offset = 0

        while len(songs) < limit:
            try:
                if search_type == "genre":
                    results = sp.search(q=f"genre:{query}", type="track", limit=min(50, limit - len(songs)), offset=offset)
                elif search_type == "year":
                    results = sp.search(q=f"year:{query}", type="track", limit=min(50, limit - len(songs)), offset=offset)
                elif search_type == "artist":
                    results = sp.search(q=f"artist:{query}", type="track", limit=min(50, limit - len(songs)), offset=offset)
                else:
                    raise ValueError("Invalid search_type. Choose from 'genre', 'year', or 'artist'.")

                tracks = results.get('tracks', {}).get('items', [])

                if not tracks:
                    break

                for track in tracks:
                    song_name = track['name']
                    artist_name = ', '.join(artist['name'] for artist in track['artists'])

                    # Fetch additional genres from artist information
                    artist_ids = [artist['id'] for artist in track['artists']]  # Get artist IDs

                    # Construct song dictionary with additional genres
                    song_info = {
                        'song_id': track['id'],
                        'song_name': song_name,
                        'artist': artist_name,
                        'album': track['album']['name'],
                        'duration_ms': track['duration_ms'],
                        'popularity': track['popularity'],
                        'genre_name':query,
                        'release_year': track['album']['release_date'][:4],
                        'spotify_link': track['external_urls']['spotify']
                    }
                    songs.append(song_info)

                offset += len(tracks)
            except Exception as e:
                print(f"Error fetching songs for query {query}: {e}")
                break

        data.extend(fn_fetch_lyrics_concurrently(songs))

    df = pd.DataFrame(data)
    return df


def fn_clean_text(lyrics):
    if not isinstance(lyrics, str):  # Handle non-string or None inputs
        return ""

    stop_words = set(stopwords.words("english"))
    custom_stopwords = stop_words - {"not", "no", "never"}
    lemmatizer = WordNetLemmatizer()

    # Step 1: Normalize text (lowercase, remove extra spaces, expand contractions)
    lyrics = "\n".join([contractions.fix(line.lower()) for line in lyrics.splitlines() if line.strip()])

    # Step 2: Remove punctuations and normalize repeated characters
    lyrics = re.sub(r"[^\w\s]", "", lyrics)  # Remove all non-alphanumeric characters (including emojis)
    lyrics = re.sub(r"(.)\1{2,}", r"\1", lyrics)  # Normalize repeated characters (e.g., "loooove" -> "love")

    # Step 3: Remove numbers
    lyrics = re.sub(r"\d+", "", lyrics)

    # Step 4: Normalize whitespace
    lyrics = re.sub(r"\s+", " ", lyrics).strip()

    return lyrics


def fn_process_text(lyrics):
    stop_words = set(stopwords.words("english"))
    custom_stopwords = stop_words - {"not", "no", "never"}
    lemmatizer = WordNetLemmatizer()

    # Step 1: Tokenize, remove stopwords, and lemmatize
    unique_lines = list(dict.fromkeys(lyrics.splitlines()))
    cleaned_lines = []
    for line in unique_lines:
        tokens = line.split()

        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords and len(word) > 2]

        # Filter by POS tags
        tokens = [word for word, pos in pos_tag(tokens) if pos.startswith(('JJ', 'VB', 'NN', 'RB'))]

        cleaned_lines.append(" ".join(tokens))

    # Step 2: Remove non-ASCII characters (if any remain)
    processed_lyrics = " ".join(cleaned_lines)
    processed_lyrics = processed_lyrics.encode("ascii", "ignore").decode()

    return processed_lyrics


# Load the Hugging Face pipeline for emotion detection
def fn_initialize_mood_detector():
    """
    Initializes the Hugging Face pipeline using the cardiffnlp/twitter-roberta-base-emotion model.

    Returns:
        pipeline: Hugging Face text-classification pipeline.
    """
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-emotion",
        top_k=True  
    )

def fn_detect_mood(lyrics, mood_detector):
    """
    Detects the mood of the song and provides emotion scores.

    Args:
        lyrics (str): Lyrics of the song.
        mood_detector (pipeline): Hugging Face pipeline for text classification.

    Returns:
        list: Detected emotions and their scores.
    """
    try:
        # Truncate lyrics to avoid processing errors
        truncated_lyrics = lyrics[:5000] 
        result = mood_detector(truncated_lyrics)
        return result[0] if result else [{"label": "Unknown", "score": 0.0}]
    except Exception as e:
        return [{"label": "Error", "score": 0.0}]


