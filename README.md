# AI-Powered Song Popularity Prediction & Recommendation System

## Overview
This project is an AI-driven system that integrates **machine learning, natural language processing, and recommendation algorithms** to analyze song lyrics, predict popularity, and generate personalized song recommendations based on moods. It consists of multiple components that enable data collection, processing, model training, and a user-friendly interface for interacting with the system.

## Features
- **Lyrics Dataset Generation:** Fetches song lyrics and metadata from **Spotify** and other sources, processes text, and stores structured data in a database.
- **Mood-Based Song Recommendation:** Suggests songs based on user-selected moods using sentiment analysis and similarity-based filtering.
- **AI Songwriting Assistant:** Generates AI-assisted song lyrics based on user prompts and inspirations.
- **Popularity Prediction Model:** Uses **FastText embeddings** and **Random Forest Regression** to predict a song's popularity based on its lyrics.
- **Retrieval-Augmented Generation (RAG):** Enhances AI-generated lyrics with similar song references to improve creativity and coherence.

---

## Project Structure
```
├── ai_sw_recommender_main.py             # Streamlit-based web application for song recommendation & lyric generation
├── ai_songwriter_song_recommender.py     # Core functions for AI-powered song recommendations and lyric generation
├── lyrics_dataset_generator_main.py      # Script to generate and preprocess the lyrics dataset
├── popularity_prediction_model_trainer.py# ML pipeline for training a song popularity prediction model
├── spotify_lyrics_dataset_generator_helper.py # Helper functions for data collection, text processing, and emotion analysis
├── model_files                           # Folder to store trained ML models (RandomForest & FastText)
└── requirements.txt                      # List of dependencies required to run the project
```

---

## Setup & Installation
### Prerequisites
Ensure you have Python 3.8+ installed. Install required dependencies using:
```sh
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file with the following credentials:
```env
SPOTIFY_CLIENT_ID=<your_spotify_client_id>
SPOTIFY_CLIENT_SECRET=<your_spotify_client_secret>
SPOTIFY_DB_PASS=<your_database_password>
OPENAI_API_KEY=<your_openai_api_key>
```

### Database Setup
The project relies on **MySQL** to store lyrics and song metadata. Set up your database and configure the connection string:
```sh
mysql+mysqlconnector://ironhack:{SPOTIFY_DB_PASS}@127.0.0.1/spotify_lyrics_db
```
Ensure your database schema has a table named `dim_tracks_details` for storing processed song lyrics and metadata.

---

## Running the Project
### 1. Generate Lyrics Dataset
Run the following command to fetch lyrics and preprocess them for mood analysis:
```sh
python lyrics_dataset_generator_main.py
```

### 2. Train Popularity Prediction Model
```sh
python popularity_prediction_model_trainer.py
```
This script trains **FastText** and **RandomForestRegressor**, evaluates the model, and saves the trained models in `model_files/`.

### 3. Start the Web Application
Run the Streamlit-based web app for song recommendations and AI songwriting:
```sh
streamlit run ai_sw_recommender_main.py
```

---

## How It Works
### **1. Lyrics Dataset Generator**
- Extracts song lyrics from **Spotify** and external APIs.
- Cleans and processes lyrics by removing noise and detecting moods.
- Stores data in a MySQL database.

### **2. Popularity Prediction Model**
- Uses **FastText** embeddings to convert lyrics into numerical vectors.
- Trains a **RandomForestRegressor** to predict a song's popularity based on lyrics.
- Evaluates model performance using MAE, RMSE, and MSE.

### **3. AI-Powered Song Recommendation & Writing**
- **Mood-Based Playlist Generator**: Users select a mood, and the system recommends songs.
- **AI Songwriter with RAG**: Generates song lyrics inspired by user prompts and existing song structures. The **Retrieval-Augmented Generation (RAG)** method enhances lyric generation by incorporating similar lyrics from a curated dataset, improving coherence and relevance.
- **Popularity Prediction**: Predicts the potential popularity of AI-generated lyrics.

---

## Technologies Used
- **Machine Learning:** FastText, RandomForestRegressor
- **NLP & Text Processing:** Hugging Face, NLTK, OpenAI GPT-4
- **Database:** MySQL, SQLAlchemy
- **Web Framework:** Streamlit
- **Data Collection:** Spotipy (Spotify API), Lyrics.ovh API

---

## Future Improvements
- Improve popularity prediction with deep learning models like **LSTMs** or **Transformer-based models**.
- Expand dataset to include additional song features like genre, tempo, and user engagement metrics.
- Enhance AI-generated lyrics using **fine-tuned LLMs**.
- Implement a **user feedback loop** to refine recommendations over time.

---

## Contributors
- **Avirup Chakraborty** - Developer & Data Engineer
- **LinkedIn** - https://www.linkedin.com/in/avirup-chakraborty/
If you have any questions or suggestions, feel free to reach out!

