from modules import spotify_lyrics_dataset_generator_helper as sld
from modules import ai_songwriter_song_recommender as ais
import streamlit as st


# Streamlit Application
st.title("🎵 A.I. Creative Assistant")
st.write("Explore songs by mood or generate your own song lyrics based on a mood and inspiration.")

df = ais.load_sample_dataframe()

# Sidebar
st.sidebar.title("Navigation")

app_mode = st.sidebar.radio("Choose a feature:", ["Mood-Based Playlist", "Song Writer"])

if app_mode == "Mood-Based Playlist":

    st.header("🎶 Mood-Based Playlist Generator")
    unique_moods = df['mood'].str.capitalize().unique()
    selected_mood = st.selectbox("Select a mood:", unique_moods)

    num_songs = st.slider("Number of songs to recommend:", min_value=1, max_value=10, value=5)

    col1, col2 = st.columns([5, 1])

    with col1:
        if st.button("Generate Playlist"):
            with st.spinner("⏳ Finding songs with the selected mood..."):
                recommendations = ais.recommend_songs_by_mood(selected_mood, df, num_songs)

                if not recommendations.empty:
                    st.success(f"✅ Found {len(recommendations)} songs with the mood '{selected_mood}'.")

                    for _, row in recommendations.iterrows():
                        st.markdown(f"""
                        <div style="border: 1px solid #e3e3e3; border-radius: 12px; padding: 16px; margin-bottom: 16px; background-color: #f9f9f9;">
                            <h4 style="margin: 0;">🎵 <b>{row['song_name']}</b></h4>
                            <p style="margin: 0; color: #555;">By: <i>{row['artist']}</i></p>
                            <p style="margin: 8px 0;"><b>Popularity:</b> {round(row['popularity'], 0)}</p>
                            <p style="margin: 8px 0;"><b>Mood Strength:</b> {round(row['mood_strength'],2)}</p>
                            <a href="{row['spotify_link']}" style="color: #1DB954; font-weight: bold;" target="_blank">▶️ Listen on Spotify</a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ No songs found with the mood '{selected_mood}'.")

    with col2:
        if st.button("Reset"):
            st.rerun()

elif app_mode == "Song Writer":

    st.header("📝 AI-Powered Song Writer")
    user_prompt = st.text_area("Describe your inspiration for the song:")
    detected_mood = ais.map_prompt_to_mood(user_prompt) if user_prompt else "Neutral"
    st.write(f"Detected Mood: {detected_mood.capitalize()}")

    selected_mood = st.selectbox("Select a mood:", options=[detected_mood.capitalize()] + list(df['mood'].str.capitalize().unique()))

    selected_song = st.selectbox("Select a song for inspiration:", df['song_name'].unique())
    selected_artist = df.loc[df['song_name'] == selected_song, 'artist'].values[0]
    selected_lyrics = df.loc[df['song_name'] == selected_song, 'lyrics'].values[0]

    st.write(f"**Selected Song:** {selected_song} by {selected_artist}")
    st.text_area("Inspiration Lyrics:", value=selected_lyrics, height=150)

    col1, col2 = st.columns([5, 1])

    with col1:
        if st.button("Generate Song"):
            with st.spinner("⏳ Writing your song..."):
                song_lyrics = ais.generate_song_lyrics(selected_artist, selected_song, selected_lyrics, user_prompt, selected_mood)
                st.success("✅ Your song has been created!")
                st.markdown(f"""
                <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 20px; background-color: #f7fcff;">
                    <h3 style="text-align: center; color: #1DB954;">🎤 Your Generated Song Lyrics</h3>
                    <div style="font-family: 'Courier New', Courier, monospace; font-size: 16px; white-space: pre-wrap; line-height: 1.5;">
                        {"<br>".join(song_lyrics.splitlines())}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if song_lyrics and song_lyrics != "An error occurred while generating the song lyrics.":
                    clean_song_lyrics = sld.fn_clean_text(song_lyrics)
                    clean_song_lyrics = sld.fn_process_text(clean_song_lyrics)
                    popularity_score = ais.predict_song_popularity(clean_song_lyrics)
                    st.markdown(f"""<div style="border: 2px solid #1DB954; border-radius: 12px; padding: 16px; background-color: #e8f5e9; text-align: center;">
                                    <h3 style="color: #1DB954; margin: 0;">🎯 Predicted Popularity Score</h3>
                                    <p style="font-size: 24px; font-weight: bold; margin: 0; color: #2e7d32;">{int(popularity_score)} / 100</p>
                                </div>
                                """, unsafe_allow_html=True)

                    st.download_button(
                        label="📥 Download Song Lyrics",
                        data=song_lyrics,
                        file_name="{}.txt".format(song_lyrics.splitlines()[0]),
                        mime="text/plain"
                    )

    with col2:
        if st.button("Reset"):
            st.rerun()
