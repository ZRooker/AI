from flask import Flask, render_template, request, session,redirect, url_for

app = Flask(__name__)
app.secret_key = 'XXXXXXX' #Change according to needs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/username', methods=['GET', 'POST'])
def username():
    if request.method == 'POST':
        username = request.form['username']
        session['username'] = username
        return render_template('home.html', username=username)
    else:
        return render_template('index.html')

## Getting the home.html button pages upon request
@app.route('/song_search')
def song_search():
    return render_template('song_search.html')

# ## Getting the home.html button pages upon request
@app.route('/playlist_search')
def playlist_search():
    playlist_df = user_playlist()
    return render_template('playlist_search.html', playlist_df=playlist_df)


## Getting the home.html button pages upon request
@app.route('/user')
def user():
    user = user_Recommendation()
    print(user)
    session['user_fordelete2'] = user.to_json()
    return render_template('user.html', playlist = user)


## Getting input from user for song then searching for song in my python code
@app.route('/search_song', methods=['GET', 'POST'])
def songname():
    if request.method == 'POST':
        songname = request.form['songname']
        print(songname)
        sp = spotify_verification()
        recommendations = song_Recommendation(songname)
        print(recommendations)
        return render_template('song.html', songs=recommendations)
    else:
        return render_template('index.html')

## Getting input from user for playlist then searching for playlist in my python code
@app.route('/search_userplaylist', methods=['GET', 'POST'])
def search_userplaylist():
    if request.method == 'POST':
        playlistname = request.form['playlistname']
        print(playlistname)
        u = user_playlist()
        ID=find_playlist(playlistname,u)
        playlist = playlist_Recommendation(ID)
        print(playlist)
        session['playlist_fordelete'] = playlist.to_json()

        # keep_playlist(playlist, input)
        # input("Do you want to keep all playlists, delete all playlists, or keep a specific playlist? (Enter 'all', 'none', or a cluster number): ")
        return render_template('playlist.html', playlist = playlist)
    else:
        return render_template('index.html')

#Getting input on which playlist to delete this is for the playlist section 
@app.route('/playlist_delete', methods=['GET', 'POST'])
def playlist_delete():
    if request.method == 'POST':
        deletename = request.form['deletename']
        print(deletename)
        
        deleted = keep_playlist(pd.read_json(session['playlist_fordelete']), deletename)
        if deleted is None:
            return redirect(url_for('playlist_search'))
        else:
            return render_template('playlist.html', playlist=deleted)
    else:
        return render_template('index.html')

#Getting input on which playlist to delete this is for the user section 
@app.route('/user_delete', methods=['GET', 'POST'])
def user_delete():
    if request.method == 'POST':
        userdeletename = request.form['userdeletename']
        print(userdeletename)
        
        deleted = keep_playlist(pd.read_json(session['user_fordelete2']), userdeletename)
        if deleted is None:
            return redirect(url_for('home.html'))
        else:
            return render_template('user.html', playlist=deleted)
    else:
        return render_template('index.html')


##Machine Learning Code 
 
import os
import json
#!pip install spotipy
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd
import numpy as np
import seaborn as sns 
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
    cid ='XXXXXXXXXXXXXXXXXXXXXXXXXX' # Client ID; copy this from your app created on beta.developer.spotify.com
    secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXX' # Client Secret; copy this from your app
    username = session['username'] # Your Spotify username

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

#Deletes the playlist based off user input
def keep_playlist(playlist_df, action):
    sp = spotify_verification()
    if action == 'all':
        print("All playlists will be kept.")
        return playlist_df
    elif action == 'none':
        for playlist_id in playlist_df['id']:
            sp.user_playlist_unfollow(user=username, playlist_id=playlist_id)
        print("All playlists have been deleted.")
        return None
    elif action.isdigit() and 0 <= int(action) < len(playlist_df):
        cluster_id = int(action)
        playlist_id = playlist_df.iloc[cluster_id]['id']
        print(f"Playlist '{playlist_df.iloc[cluster_id]['Name']}' will be kept.")
        playlist_df = playlist_df.iloc[[cluster_id]]
        return playlist_df
    else:
        print(f"Invalid input. Please enter 'all', 'none', or a cluster number (0-{len(playlist_df)-1}).")

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

    # Get the cluster labels for each data point
    labels = kmeans.labels_
    df_recom['clusters'] = labels
    
    cluster_0_data = df_recom[df_recom['clusters']==0]
    cluster_1_data = df_recom[df_recom['clusters']==1]
    cluster_2_data = df_recom[df_recom['clusters'] == 2]
    cluster_3_data = df_recom[df_recom['clusters']==3]

    user_id = session['username']

    # Define a dictionary to hold the cluster playlists
    cluster_playlists = {}

    # Loop through the clusters and create a new playlist for each one
    for cluster_id in range(4):
        playlist_name = f'Playlist Cluster {cluster_id}'
        playlist = sp.user_playlist_create(user=user_id, name=playlist_name)
        cluster_playlists[cluster_id] = playlist['id']

    # Loop through the songs in each cluster and add them to their corresponding playlist
    for cluster_id, cluster_data in enumerate([cluster_0_data, cluster_1_data, cluster_2_data, cluster_3_data]):
        for _, song_data in cluster_data.iterrows():
            song_id = song_data['iD']
            track_info = sp.track(song_id)
            track_uri = track_info['uri']
            sp.user_playlist_add_tracks(user=user_id, playlist_id=cluster_playlists[cluster_id], tracks=[track_uri])

        # Define a list to hold the playlist information
    playlist_info = []

    # Loop through the cluster playlists and add the playlist ID and name to the list
    for cluster_id in range(4):
        playlist_id = cluster_playlists[cluster_id]
        playlist_name = f'Playlist Cluster {cluster_id}'
        playlist_info.append({'Name': playlist_name, 'id': playlist_id})

    # Create a DataFrame from the playlist information
    playlist_df = pd.DataFrame(playlist_info)

    return playlist_df

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

    cluster_0_data = df_recom[df_recom['clusters']==0]
    cluster_1_data = df_recom[df_recom['clusters']==1]
    cluster_2_data = df_recom[df_recom['clusters'] == 2]
    cluster_3_data = df_recom[df_recom['clusters']==3]

    user_id = session['username']

    # Define a dictionary to hold the cluster playlists
    cluster_playlists = {}

    # Loop through the clusters and create a new playlist for each one
    for cluster_id in range(4):
        playlist_name = f'User Cluster {cluster_id}'
        playlist = sp.user_playlist_create(user=user_id, name=playlist_name)
        cluster_playlists[cluster_id] = playlist['id']

    # Loop through the songs in each cluster and add them to their corresponding playlist
    for cluster_id, cluster_data in enumerate([cluster_0_data, cluster_1_data, cluster_2_data, cluster_3_data]):
        for _, song_data in cluster_data.iterrows():
            song_id = song_data['iD']
            track_info = sp.track(song_id)
            track_uri = track_info['uri']
            sp.user_playlist_add_tracks(user=user_id, playlist_id=cluster_playlists[cluster_id], tracks=[track_uri])

        # Define a list to hold the playlist information
    playlist_info = []

    # Loop through the cluster playlists and add the playlist ID and name to the list
    for cluster_id in range(4):
        playlist_id = cluster_playlists[cluster_id]
        playlist_name = f'User Cluster {cluster_id}'
        playlist_info.append({'Name': playlist_name, 'id': playlist_id})

    # Create a DataFrame from the playlist information
    playlist_df = pd.DataFrame(playlist_info)

    return playlist_df

def main():
    rec_type = input("Recommendations for a song (S), playlist (P), or user data (U)? ")

    if rec_type == 'S':
        song_name = input("Search a song for recommendations: ")
        recommendations = song_Recommendation(song_name)
        print(recommendations)

    elif rec_type == 'P':
        #Getting A List of the Users Playlist
        playtype = input("User Playlist (U) or Link Playlist (L)?")
        
        if playtype == 'U': 
            u=user_playlist()
            playlist_name = input("Pick a Playlist:  ")
            
            #Using Input to find the Playlist and get the ID
            ID=find_playlist(playlist_name,u)
            playlist = playlist_Recommendation(ID)
            print(playlist)
        
        elif playtype == 'L':
            link = input('Provide Link to playlist here:')
            playlist2 = playlist_Recommendation(link)
            print(playlist2)
        
        else: 
            print('Incorrect Choice, Please Choose Again ')

    elif rec_type == 'U':
        print("Getting User History Recommendations ")
        user = user_Recommendation()
        print(user)

    else: 
        print('Request: ', rec_type, '   Invalid Type')

# if __name__ == '__main__':
#     main()

if __name__ == '__main__':
    app.run(debug=True,  host='0.0.0.0', port=5000)
