import pandas as pd
from modules import spotify_lyrics_dataset_generator_helper as sld
import traceback as tb
from datetime import datetime as dt
import os
from sqlalchemy import create_engine

SPOTIFY_DB = os.getenv("SPOTIFY_DB_PASS")

#Setup the database engine
connection_string = "mysql+mysqlconnector://ironhack:{}@127.0.0.1/spotify_lyrics_db".format(SPOTIFY_DB)
engine = create_engine(connection_string)


#MAIN FUNCTION
if __name__ == '__main__':

    try:

            start_time = dt.now()
            print("Lyrics Dataset Generator Script Start Time : {}".format(start_time))

            ################################################################################
            ## Data Loading and Cleaning
            ################################################################################

            # List of genres
            query = "SELECT genre_id, genre_name from dim_genre;"

            genre_df = pd.read_sql(query, con=engine)

            popular_genres = genre_df.genre_name.tolist()


            #List of years
            #years = list(range(1920, 2025))

            # Number of songs per search query
            num_songs = 1000

            # Fetch songs with lyrics for all genres
            songs_with_lyrics_df = sld.fn_fetch_songs_with_lyrics(search_type='genre',search_queries=popular_genres, limit=num_songs)

            #Remove the rows where lyrics is not found
            songs_with_lyrics_df = songs_with_lyrics_df[songs_with_lyrics_df['lyrics'] != 'Lyrics not found']

            #Track specific information
            dim_tracks_details_df = songs_with_lyrics_df[['song_id', 
                                                          'song_name', 
                                                          'artist', 
                                                          'album', 
                                                          'duration_ms', 
                                                          'release_year', 
                                                          'spotify_link', 
                                                          'lyrics',
                                                          'popularity']].drop_duplicates().reset_index(drop=True)

            #Track genre map information
            map_tracks_genre_df = songs_with_lyrics_df[['song_id', 'genre_name']].drop_duplicates().reset_index(drop=True)

            #Map it to the id of the genre
            map_tracks_genre_df = pd.merge(map_tracks_genre_df,genre_df, on='genre_name', how='inner')
            map_tracks_genre_df = map_tracks_genre_df[['song_id','genre_id']]


            # Clean the lyrics
            dim_tracks_details_df['lyrics'] = dim_tracks_details_df['lyrics'].apply(lambda x:sld.fn_clean_text(x))

            # Process the lyrics
            dim_tracks_details_df['processed_lyrics'] = dim_tracks_details_df['lyrics'].apply(lambda x:sld.fn_process_text(x))

            # Initialize the mood detector
            mood_detector = sld.fn_initialize_mood_detector()

            # Detect emotions and scores
            dim_tracks_details_df['emotion_scores'] = dim_tracks_details_df['processed_lyrics'].apply(lambda x: sld.fn_detect_mood(x, mood_detector))

            # Extract the top emotion label
            dim_tracks_details_df['mood'] = dim_tracks_details_df['emotion_scores'].apply(lambda x: x[0]['label'] if x else "Unknown")

            # Extract the top emotion score
            dim_tracks_details_df['mood_strength'] = dim_tracks_details_df['emotion_scores'].apply(lambda x: x[0]['score'] if x else 0.0)

            # Select specific columns
            dim_tracks_details_df = dim_tracks_details_df[['song_id', 'song_name', 'artist', 'album', 'duration_ms', 
                                                          'release_year',  'spotify_link', 'lyrics', 'processed_lyrics','mood', 'mood_strength','popularity']]

            # Remove the erronoeus moods
            error_tracks = dim_tracks_details_df[dim_tracks_details_df['mood'].isin(['Error','Unknown'])]['song_id'].tolist()
            map_tracks_genre_df = map_tracks_genre_df[~map_tracks_genre_df['song_id'].isin(error_tracks)]
            dim_tracks_details_df = dim_tracks_details_df[~dim_tracks_details_df['song_id'].isin(error_tracks)]


            #Save the data into a database table
            dim_tracks_details_df.to_sql(name="dim_tracks_details", con=engine, if_exists="replace", index=False)
            map_tracks_genre_df.to_sql(name="map_tracks_genre", con=engine, if_exists="replace", index=False)

            end_time = dt.now()
            print("Lyrics Dataset Generator Script Completed:{}".format(end_time))
            duration = end_time - start_time
            td_mins = int(round(duration.total_seconds() / 60))
            print('The difference is approx. %s minutes' % td_mins)
     
    except Exception as e:
        
        error = "Lyrics Dataset Generator Script Failure :: Error Message {} with Traceback Details: {}".format(e,tb.format_exc())        
        print(error)

