import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def recommendation(cl_df:pd.DataFrame, cluster_col:str, title:str, artist:str):
    """ This is a method that recommends similar songs 
                Args:
                    cl_df: list of songs information and clustering results
                    cluster_col: column name of clustering information 
                    title: title of song
                    artist: name of the artist for the song
                Returns:
                    result: title and artist name of songs that is simlar 
                    suggestions: index for the title and artist name of songs that is simlar
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
    """ This is a method that recommends similar songs 
                Args:
                    title: title of song
                    artist: name of the artist for the song
                Returns:
                    result: title and artist name of songs that is simlar 
                    suggestions: index for the title and artist name of songs that is simlar
    """
    radar_chart_data =data.loc[:,:'popularity']
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