# 🎵 AI Songwriter & Recommender

## Overview
This project is an AI-powered music assistant that provides song recommendations based on mood and generates song lyrics inspired by user input. The system leverages machine learning models, natural language processing, and the Spotify API to analyze lyrics, detect moods, and predict song popularity.

## Features
- 🎶 **Mood-Based Playlist Generator**: Recommends songs based on mood.
- 📝 **AI-Powered Songwriting**: Generates lyrics inspired by user input.
- 📊 **Popularity Prediction**: Predicts the popularity of generated lyrics.
- 🎼 **Lyrics Dataset Generator**: Extracts and processes song lyrics from Spotify.

## Technologies Used
- **Python**: Core programming language
- **Streamlit**: Web UI for the interactive application
- **Scikit-learn & XGBoost**: Machine learning for popularity prediction
- **Gensim (FastText)**: Word embeddings for text processing
- **OpenAI API**: AI-powered lyric generation
- **Hugging Face Transformers**: Mood detection from lyrics
- **MySQL & SQLAlchemy**: Database management for song datasets
- **Spotipy**: Spotify API integration for song metadata retrieval

## Project Structure
```
├── ai_sw_recommender_main.py         # Streamlit app for song recommendation and lyric generation
├── popularity_prediction_model_trainer.py  # Model training for song popularity prediction
├── lyrics_dataset_generator_main.py  # Fetches & processes song lyrics dataset
├── model_files/                      # Stored ML models (FastText & RandomForest)
├── modules/                          # Stored functions which are called from the apps
  ├── spotify_lyrics_dataset_generator_helper.py # Spotify API helper functions
  ├── ai_songwriter_song_recommender.py # Song recommendation logic & OpenAI API integration
├── README.md                         # Project documentation
├── requirements.txt                   # Required dependencies
```
### Setup the MySQl Database
  1. Install MySQL in your local machine
  2. Create a user ironhack
  3. Create a database spotify_lyrics_db
  4. Create the tables given in the Table DDL script  

## Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ai-songwriter-recommender.git
cd ai-songwriter-recommender
```

### 2️⃣ Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_DB_PASS=your_mysql_password
```

### 5️⃣ Run the Application
```bash
streamlit run ai_sw_recommender_main.py
```

## Usage
### 🎵 Mood-Based Playlist
1. Select the **Mood-Based Playlist** option.
2. Choose a mood from the dropdown.
3. Generate a playlist based on the selected mood.

### 📝 AI-Powered Songwriting
1. Select the **Song Writer** option.
2. Enter a prompt describing your inspiration.
3. Select a song & artist for lyrical inspiration.
4. Generate song lyrics with AI.
5. View the predicted popularity score.
6. Download your generated song.

## Model Training
To train the popularity prediction model:
```bash
python popularity_prediction_model_trainer.py
```

