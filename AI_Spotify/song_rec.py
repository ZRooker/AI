import os
import json
#!pip install spotipy
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle


### Supporting Functions ###

#Verifying the user in spotify
def spotify_verification():
    cid ='d8762235995b4ddcbc0df90739df9a63' # Client ID; copy this from your app created on beta.developer.spotify.com
    secret = '887a4eccdbde4195bfc0f7f4c472fe70' # Client Secret; copy this from your app
    username = '12155311349' # Your Spotify username

    #for avaliable scopes see https://developer.spotify.com/web-api/using-scopes/
    scope = 'user-library-read playlist-modify-public playlist-read-private'
    redirect_uri='https://localhost:8080/callback/'
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    token = util.prompt_for_user_token(username,scope,client_id=cid,client_secret=secret,redirect_uri="https://localhost:8080/callback/")
    if token:
        sp = spotipy.Spotify(auth=token)
    else:
        print("Can't get token for", username)
    return sp

#Gets a list of users playlists 
def user_playlist():
    sp = spotify_verification()
    # get the user's playlists
    playlists = sp.current_user_playlists()

    # create a DataFrame with the playlist names and IDs
    playlist_df = pd.DataFrame([(playlist['name'], playlist['id']) for playlist in playlists['items']],
                            columns=['name', 'id'])

    # print the DataFrame
    print('Your Playlists, Please Choose one: ')
    print(playlist_df)
    return playlist_df

#Finds the id associated with a playlist
def find_playlist(playlist_name,playlist_df):
    # find the playlist ID that matches the user's input
    matching_playlists = playlist_df[playlist_df['name'].str.contains(playlist_name, case=False)]
    if len(matching_playlists) == 0:
        print("No matching playlists found.")
    else:
        playlist_id = matching_playlists.iloc[0]['id']
        print(f"Selected playlist: {matching_playlists.iloc[0]['name']} (ID: {playlist_id})")
    return playlist_id

### Recommendation Functions ###

#Gives recommendations based off a song 
def song_Recommendation(song_name):
    sp = spotify_verification()
    # Define the search query
    search_results = sp.search(q=song_name, type='track', limit=1)
    track_id = search_results['tracks']['items'][0]['id']

    # Get the audio features, year, and popularity of the song
    track_id = search_results['tracks']['items'][0]['id']
    audio_features = sp.audio_features([track_id])[0]
    year = search_results['tracks']['items'][0]['album']['release_date'][:4]
    popularity = search_results['tracks']['items'][0]['popularity']

    # Find similar songs based on audio features, year, and popularity
    similar_songs = sp.recommendations(seed_tracks=[track_id], target_popularity=popularity, limit=100, **audio_features)
    # similar_songs = sp.recommendations(seed_tracks=[track_id], limit=100, **audio_features)

    # Extract relevant info from recommendations and store in DataFrame
    columns = ['song name', 'artist', 'popularity', 'year', 'iD', 'danceability', 'energy','loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    data = []
    for i, track in enumerate(similar_songs['tracks']):
        name = track['name']
        artist = track['artists'][0]['name']
        popularity = track['popularity']
        year = track['album']['release_date'][:4]
        track_id = track['id']
        audio_features = sp.audio_features([track_id])[0]
        danceability = audio_features['danceability']
        loudness = audio_features['loudness']
        energy = audio_features['energy']
        speechiness = audio_features['speechiness']
        acousticness = audio_features['acousticness']
        instrumentalness = audio_features['instrumentalness']
        liveness = audio_features['liveness']
        valence = audio_features['valence']
        tempo = audio_features['tempo']
        data.append([name, artist, popularity, year, track_id, danceability, loudness,energy, speechiness, acousticness, instrumentalness, liveness, valence, tempo])
    df = pd.DataFrame(data, columns=columns)

    # Scale between 0 and 1
    df["tempo"] = (df["tempo"] / df["tempo"].max())
    df["loudness"] = (df["loudness"] / df["loudness"].max())
    df["energy"] = (df["energy"] / df["energy"].min())
    df["popularity"] = (df["popularity"] / 100)

    from sklearn.neighbors import NearestNeighbors

    # Define the features to use for the model
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    # Create a new dataframe with only the features to use for the model
    df_model = df[features]

    # Fit the k-nearest neighbors model using the new dataframe
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(df_model)

    # Find the k nearest neighbors for the user-requested song
    song_data = df[df['iD'] == track_id][features]
    distances, indices = knn.kneighbors(song_data)

    # Create a new dataframe with the nearest neighbors and their features
    neighbor_data = df.iloc[indices[0]]

    return neighbor_data

#Gives recommendations based off a playlist 
def playlist_Recommendation(playlist):
    sp = spotify_verification()
    playlist = sp.playlist(playlist)

    # Get the audio features for each track in the playlist
    audio_features = {}
    for track in playlist['tracks']['items']:
        audio_features[track['track']['id']] = sp.audio_features(track['track']['id'])[0]

    # Create a dataframe with the track information
    columns = ['song name', 'artist', 'popularity', 'release date', 'iD', 'danceability', 'energy','loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    data = []
    for i, track in enumerate(playlist['tracks']['items']):
        name = track['track']['name']
        artist = track['track']['artists'][0]['name']
        popularity = track['track']['popularity']
        year = track['track']['album']['release_date'][:4]
        track_id = track['track']['id']
        audio_features = sp.audio_features([track_id])[0]
        if audio_features is None:
            danceability = 0
            loudness = 0
            energy = 0
            speechiness = 0
            acousticness = 0
            instrumentalness = 0
            liveness = 0
            valence = 0
            tempo = 0
        else:
            danceability = audio_features['danceability']
            loudness = audio_features['loudness']
            energy = audio_features['energy']
            speechiness = audio_features['speechiness']
            acousticness = audio_features['acousticness']
            instrumentalness = audio_features['instrumentalness']
            liveness = audio_features['liveness']
            valence = audio_features['valence']
            tempo = audio_features['tempo']
        data.append([name, artist, popularity, year, track_id, danceability, loudness,energy, speechiness, acousticness, instrumentalness, liveness, valence, tempo])
    df = pd.DataFrame(data, columns=columns)
    # Scale between 0 and 1
    df["tempo"] = (df["tempo"] / df["tempo"].max())
    df["loudness"] = (df["loudness"] / df["loudness"].max())
    df["energy"] = (df["energy"] / df["energy"].min())
    df["popularity"] = (df["popularity"] / 100)
    
    recom = []

    for index, row in df.iterrows():
        # Find similar songs based on audio features, year, and popularity
        similar_songs = sp.recommendations(seed_tracks=[row['iD']], target_popularity=popularity, limit=5, **audio_features)
        recom.append(similar_songs)
            
    # Extract relevant info from recommendations and store in DataFrame
    columns = ['name', 'artist', 'popularity', 'year', 'iD', 'danceability', 'energy','loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    data = []
    for i, track_list in enumerate(recom):
        for track in track_list['tracks']:
            name = track['name']
            artist = track['artists'][0]['name']
            popularity = track['popularity']
            year = track['album']['release_date'][:4]
            track_id = track['id']
            audio_features = sp.audio_features([track_id])[0]
            if audio_features is None:
                danceability = 0
                loudness = 0
                energy = 0
                speechiness = 0
                acousticness = 0
                instrumentalness = 0
                liveness = 0
                valence = 0
                tempo = 0
            else:
                danceability = audio_features['danceability']
                loudness = audio_features['loudness']
                energy = audio_features['energy']
                speechiness = audio_features['speechiness']
                acousticness = audio_features['acousticness']
                instrumentalness = audio_features['instrumentalness']
                liveness = audio_features['liveness']
                valence = audio_features['valence']
                tempo = audio_features['tempo']
            data.append([name, artist, popularity, year, track_id, danceability, loudness,energy, speechiness, acousticness, instrumentalness, liveness, valence, tempo])
    df_recom = pd.DataFrame(data, columns=columns)

    # Scale between 0 and 1
    df_recom["tempo"] = (df_recom["tempo"] / df_recom["tempo"].max())
    df_recom["loudness"] = (df_recom["loudness"] / df_recom["loudness"].max())
    df_recom["energy"] = (df_recom["energy"] / df_recom["energy"].min())
    df_recom["popularity"] = (df_recom["popularity"] / 100)
    df_recom = df_recom.drop_duplicates(subset=['name'])

    # Define the features to use for the model
    features = ['year','danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    # Create a new dataframe with only the features to use for the model
    df_model = df_recom[features]

    # Scale the features
    scaler = StandardScaler()
    df_model_scaled = scaler.fit_transform(df_model)

    # Fit the K-means model using the new dataframeabs
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(df_model_scaled)

    # Export the K-means model
    with open('kmeans_model.pkl', 'wb') as file:
        pickle.dump(kmeans, file)

    # Get the cluster labels for each data point
    labels = kmeans.labels_
    df_recom['clusters'] = labels
    # filter the data to only include cluster 2 values
    cluster_0_data = df_recom[df_recom['clusters']==0]
    cluster_1_data = df_recom[df_recom['clusters']==1]
    cluster_2_data = df_recom[df_recom['clusters'] == 2]
    cluster_3_data = df_recom[df_recom['clusters']==3]

    return cluster_0_data,cluster_1_data,cluster_2_data,cluster_3_data

#Gets user recommendations based off their listening history
def user_Recommendation():
    sp = spotify_verification()
    short_term_tracks = sp.current_user_top_tracks(time_range='short_term', limit=50)
    medium_term_tracks = sp.current_user_top_tracks(time_range='medium_term', limit=50)

    # Merge the results and remove duplicates
    all_tracks = short_term_tracks['items'] + medium_term_tracks['items']
    unique_tracks = {track['id']: track for track in all_tracks}.values()

    # Get the audio features for each track
    audio_features = {}
    for track in unique_tracks:
        audio_features[track['id']] = sp.audio_features(track['id'])[0]

    # Create a dataframe with the track information
    columns = ['song name', 'artist', 'popularity', 'release date', 'iD', 'danceability', 'energy','loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    data = []
    for i, track in enumerate(unique_tracks):
        name = track['name']
        artist = track['artists'][0]['name']
        popularity = track['popularity']
        year = track['album']['release_date'][:4]
        track_id = track['id']
        audio_features = sp.audio_features([track_id])[0]
        if audio_features is None:
            danceability = 0
            loudness = 0
            energy = 0
            speechiness = 0
            acousticness = 0
            instrumentalness = 0
            liveness = 0
            valence = 0
            tempo = 0
        else:
            danceability = audio_features['danceability']
            loudness = audio_features['loudness']
            energy = audio_features['energy']
            speechiness = audio_features['speechiness']
            acousticness = audio_features['acousticness']
            instrumentalness = audio_features['instrumentalness']
            liveness = audio_features['liveness']
            valence = audio_features['valence']
            tempo = audio_features['tempo']
        data.append([name, artist, popularity, year, track_id, danceability, loudness,energy, speechiness, acousticness, instrumentalness, liveness, valence, tempo])
    df = pd.DataFrame(data, columns=columns)

    # Scale between 0 and 1
    df["tempo"] = (df["tempo"] / df["tempo"].max())
    df["loudness"] = (df["loudness"] / df["loudness"].max())
    df["energy"] = (df["energy"] / df["energy"].min())
    df["popularity"] = (df["popularity"] / 100)

    recom = []

    for index, row in df.iterrows():
        # Find similar songs based on audio features, year, and popularity
        similar_songs = sp.recommendations(seed_tracks=[row['iD']], target_popularity=popularity, limit=5, **audio_features)
        recom.append(similar_songs)
            
    # Extract relevant info from recommendations and store in DataFrame
    columns = ['name', 'artist', 'popularity', 'year', 'iD', 'danceability', 'energy','loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    data = []
    for i, track_list in enumerate(recom):
        for track in track_list['tracks']:
            name = track['name']
            artist = track['artists'][0]['name']
            popularity = track['popularity']
            year = track['album']['release_date'][:4]
            track_id = track['id']
            audio_features = sp.audio_features([track_id])[0]
            if audio_features is None:
                danceability = 0
                loudness = 0
                energy = 0
                speechiness = 0
                acousticness = 0
                instrumentalness = 0
                liveness = 0
                valence = 0
                tempo = 0
            else:
                danceability = audio_features['danceability']
                loudness = audio_features['loudness']
                energy = audio_features['energy']
                speechiness = audio_features['speechiness']
                acousticness = audio_features['acousticness']
                instrumentalness = audio_features['instrumentalness']
                liveness = audio_features['liveness']
                valence = audio_features['valence']
                tempo = audio_features['tempo']
            data.append([name, artist, popularity, year, track_id, danceability, loudness,energy, speechiness, acousticness, instrumentalness, liveness, valence, tempo])
    df_recom = pd.DataFrame(data, columns=columns)

    # Scale between 0 and 1
    df_recom["tempo"] = (df_recom["tempo"] / df_recom["tempo"].max())
    df_recom["loudness"] = (df_recom["loudness"] / df_recom["loudness"].max())
    df_recom["energy"] = (df_recom["energy"] / df_recom["energy"].min())
    df_recom["popularity"] = (df_recom["popularity"] / 100)
    df_recom = df_recom.drop_duplicates(subset=['name'])
    # Define the features to use for the model
    features = ['year','danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    # Create a new dataframe with only the features to use for the model
    df_model = df_recom[features]

    # Scale the features
    scaler = StandardScaler()
    df_model_scaled = scaler.fit_transform(df_model)

    # Fit the K-means model using the new dataframeabs
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(df_model_scaled)

    # Get the cluster labels for each data point
    labels = kmeans.labels_
    df_recom['clusters'] = labels

    # filter the data to only include cluster 2 values
    cluster_0_data = df_recom[df_recom['clusters']==0]
    cluster_1_data = df_recom[df_recom['clusters']==1]
    cluster_2_data = df_recom[df_recom['clusters'] == 2]
    cluster_3_data = df_recom[df_recom['clusters']==3]

    return cluster_0_data, cluster_1_data, cluster_2_data, cluster_3_data

def main():
    rec_type = input("Recommendations for a song (S), playlist (P), or user data (U)? ")

    if rec_type == 'S':
        song_name = input("Search a song for recommendations: ")
        recommendations = song_Recommendation(song_name)
        print(recommendations)

    elif rec_type == 'P':
        #Getting A List of the Users Playlist
        u=user_playlist()
        playlist_name = input("Pick a Playlist:  ")
        
        #Using Input to find the Playlist and get the ID
        ID=find_playlist(playlist_name,u)
        playlist = playlist_Recommendation(ID)
        print(playlist)

    elif rec_type == 'U':
        print("Getting User History Recommendations ")
        user = user_Recommendation()
        print(user)

    else: 
        print('Request: ', rec_type, '   Invalid Type')

if __name__ == '__main__':
    main()