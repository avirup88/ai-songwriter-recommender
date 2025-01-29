import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error
from gensim.models import FastText
import numpy as np
import sys
import traceback as tb
from datetime import datetime as dt
import joblib

# Set up the database password from environment variables
SPOTIFY_DB = os.getenv("SPOTIFY_DB_PASS")

# Setup the database engine
connection_string = "mysql+mysqlconnector://ironhack:{}@127.0.0.1/spotify_lyrics_db".format(SPOTIFY_DB)
engine = create_engine(connection_string)

# Function to get average FastText vectors
def get_fasttext_vectors(text_series, model):
    """
    Converts text data into vectors using a FastText model.

    Parameters:
    - text_series: A pandas Series of text data.
    - model: A trained FastText model.

    Returns:
    - Numpy array of vectors.
    """
    vectors = []
    for text in text_series:
        words = text.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            vectors.append(np.zeros(model.vector_size))
    return np.array(vectors)

# Function to preprocess the dataset
def preprocess_data(engine, query):
    """
    Loads and preprocesses data from the database.

    Parameters:
    - engine: SQLAlchemy database engine.
    - query: SQL query string to fetch data.

    Returns:
    - Tuple of lyrics (features) and popularity (target).
    """
    print("Loading data from the database...")
    songs_with_lyrics_df = pd.read_sql(query, con=engine)
    
    print("Cleaning data...")
    songs_with_lyrics_df = songs_with_lyrics_df.dropna(subset=['processed_lyrics', 'popularity'])
    songs_with_lyrics_df['popularity'] = pd.to_numeric(songs_with_lyrics_df['popularity'], errors='coerce')
    
    lyrics = songs_with_lyrics_df['processed_lyrics']
    popularity = songs_with_lyrics_df['popularity']
    
    print("Data preprocessing complete.")
    return lyrics, popularity

# Function to train FastText and RandomForest models
def train_model(X_train, y_train):
    """
    Trains a FastText model on lyrics and a RandomForest model on popularity prediction.

    Parameters:
    - X_train: Training data (lyrics).
    - y_train: Target data (popularity).

    Returns:
    - Trained FastText model and RandomForest model.
    """
    print("Training FastText model...")
    sentences = [lyric.split() for lyric in X_train]  # Tokenize the lyrics
    fasttext_model = FastText(sentences, vector_size=100, window=5, min_count=1, epochs=10)

    print("Transforming training data into vectors...")
    X_train_vectors = get_fasttext_vectors(X_train, fasttext_model)

    print("Training RandomForest model...")
    rf_model = RandomForestRegressor(n_estimators=1000, 
                                     min_samples_leaf = 4, 
                                     max_depth=30, 
                                     max_features='sqrt', 
                                     min_samples_split=2, 
                                     random_state=42)

    rf_model.fit(X_train_vectors, y_train)

    print("Model training complete.")
    return fasttext_model, rf_model

# Function to evaluate the model
def evaluate_model(model, fasttext_model, X_test, y_test):
    """
    Evaluates the trained Random Forest model on test data.

    Parameters:
    - model: Trained Random Forest model.
    - fasttext_model: Trained FastText model.
    - X_test: Test data (lyrics).
    - y_test: True popularity values.

    Outputs:
    - Prints Mean Absolute Error (MAE), Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
    """
    print("Evaluating model performance...")
    X_test_vectors = get_fasttext_vectors(X_test, fasttext_model)
    y_pred = model.predict(X_test_vectors)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("Mean Absolute Error: {:.2f}".format(mae))
    print("Mean Squared Error: {:.2f}".format(mse))
    print("Root Mean Squared Error: {:.2f}".format(rmse))


# MAIN FUNCTION
if __name__ == '__main__':
    try:
        start_time = dt.now()
        print("Script started at: {}".format(start_time))

        dataset_folder = './model_files'
        query = "SELECT processed_lyrics, popularity FROM dim_tracks_details;"
        
        print("Preprocessing data...")
        lyrics, popularity = preprocess_data(engine, query)
        
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(lyrics, popularity, test_size=0.2, random_state=42)
        
        print("Training models...")
        fasttext_model, rf_model = train_model(X_train, y_train)
        
        print("Evaluating models...")
        evaluate_model(rf_model, fasttext_model, X_test, y_test)
        
        print("Saving models to disk...")
        fasttext_model.save(os.path.join(dataset_folder, "fasttext_model.bin"))
        joblib.dump(rf_model, os.path.join(dataset_folder, "rf_model.pkl"))
        
        print("Models saved successfully.")
        
        end_time = dt.now()
        print("Script completed at: {}".format(end_time))
        duration = end_time - start_time
        print('Total runtime: approx. {} minutes.'.format(int(round(duration.total_seconds() / 60))))
     
    except Exception as e:
        error = "Error occurred: {}. Traceback: {}".format(e, tb.format_exc())
        print(error)
