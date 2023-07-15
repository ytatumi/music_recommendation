import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

def preprocess(df:pd.DataFrame,):
    """ This is a method that preprocess the dataset that is used for recommendation. 
                Args:
                    df: dataset that is used for recommendation.

                Returns:
                    preprocessed_df: preprocessed dataset
                   
    """
    min_max_scaler=preprocessing.MinMaxScaler()
    col = df.loc[:,"bpm":].columns
    normalized_data =min_max_scaler.fit_transform(df.loc[:,"bpm":])
    normalized_tracks_df = pd.DataFrame(normalized_data, columns=col)
    popularity_threshold=0.5
    normalized_tracks_df= normalized_tracks_df[normalized_tracks_df['popularity']>popularity_threshold]
    preprocessed_df= normalized_tracks_df.copy()
    preprocessed_df.insert(0, "title",df['title'])
    preprocessed_df.insert(1, "artist",df['artist'])
    preprocessed_df = preprocessed_df.drop_duplicates(subset=['title', 'artist'],keep='last')
    # print(f"size of dataframe:{preprocessed_df.shape}")
    preprocessed_df['artist']=preprocessed_df['artist'].str.lower()
    preprocessed_df['title']=preprocessed_df['title'].str.lower()

    return preprocessed_df


def kn_recommendation(df:pd.DataFrame,title:str, artist:str):
    """ This is a method that recommends similar songs 
                Args:
                    df: Songs' information
                    title: title of the song
                    artist: name of the artist for the song
                Returns:
                    recommended_df:information of reommended songs

    """
    df=pd.pivot_table(df, index=['title','artist'])
    model = NearestNeighbors(metric="cosine", algorithm ="brute")
    model.fit(df.values)
    df.reset_index(inplace=True)
    
    title=title.lower()
    artist = artist.lower()
    selected=df.loc[df.artist.str.contains(artist) & df.title.str.contains(title),:]

    distance, suggestions = model.kneighbors(selected.iloc[0,2:].values.reshape(1,-1),n_neighbors=10)
    result = [(df.iloc[suggestions.flatten()[i],0], df.iloc[suggestions.flatten()[i],1]) for i in range(len(suggestions.flatten()))]
    # print(distance)
    print(distance.flatten())
    print(suggestions)
    print(result)
    suggestions= suggestions.flatten()
    recommended_df = df.filter(items = suggestions, axis=0)
    column_to_move = recommended_df.pop("bpm")
    recommended_df.insert(2, "bpm", column_to_move)
    return recommended_df

def cluster_recommendation(cl_df:pd.DataFrame, cluster_col:str, title:str, artist:str):
    """ This is a method that recommends similar songs 
                Args:
                    cl_df: Songs' information and clustering results
                    cluster_col: column name of clustering information 
                    title: title of the song
                    artist: name of the artist for the song
                Returns:
                    recommended_df:information of recommended songs
                   
    """
    title=title.lower()
    artist = artist.lower()
    selected=cl_df.loc[cl_df.artist.str.contains(artist) & cl_df.title.str.contains(title),:]
    selected_gr = selected.loc[:,cluster_col].tolist()
    recommended_df = cl_df.loc[cl_df[cluster_col]== selected_gr[0],:]
    recommended_df = recommended_df.sample(5)
    recommended_df = pd.concat([selected,recommended_df])
    return  recommended_df


def recommendation_visualisation(data:pd.DataFrame):
    """ This is a visualization of recommended similar songs 
                Args:
                    data: information of recommended similar songs
                   
    """
    # radar_chart_data =data.loc[:,:'popularity']
    radar_chart_data =data.iloc[:,:12]
    radar_chart_data['ForClosing'] =radar_chart_data['bpm']

    categories = list (radar_chart_data.columns)[2:]
    categories[-1] = categories[0]
    categories

    angles = np.linspace(0, 2*np.pi, len(categories)-1, endpoint=False)
    angles = np.concatenate((angles,[angles[0]])) 

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot(111, polar=True)
    for i in range(5):
        data = radar_chart_data.iloc[i,2:].values
        ax.plot(angles, data, 'o-', linewidth=2, label=radar_chart_data.iloc[i,0].title())
        ax.fill(angles, data, alpha=0.05)

    ax.legend(bbox_to_anchor=(1.4,1.2), title="Songs")
    ax.set_thetagrids(angles*180/np.pi, labels=categories, fontsize=12, ha='center')
    ax.set_title(f"Comparison of Songs with \n  {radar_chart_data.iloc[0,0].title()}", fontsize=18, loc='left')
    ax.grid(True)    