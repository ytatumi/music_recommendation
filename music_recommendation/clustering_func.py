import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
from sklearn.metrics import silhouette_score 
from sklearn.cluster import KMeans 
from umap import UMAP
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np 
import hdbscan

def search_best_n_kmean_cluster(df:pd.DataFrame, min_n_clusters:int, max_n_clusters:int):
    '''
            Identifies the best number of clusters in KMeans.
            Args in:
                    df-dataset of features to clusters
                    min_n_clusters - minimum number of clusters 
                    max_n:clusters - max number of clusters
            Returns: 
                int: best number of cluster
    '''
    silhouette={}  
    for i in range(min_n_clusters, max_n_clusters):
        kmeans= KMeans(n_clusters=i,n_init=10).fit(df)
        silhouette[i] = silhouette_score(df,kmeans.labels_)

    best_n_cluster=max(silhouette,key= silhouette.get)
    best_silhouette=max(silhouette.values())
    print(f"best_n_cluster: {best_n_cluster}, silhouette:{best_silhouette}")
    return best_n_cluster


def run_kmean(df:pd.Series,n_cluster:int):
    '''
            Identifies the best number of clusters in KMeans.
            Args in:
                    df-dataset of features to clusters
                    n_cluster :number of clusters
            Returns: 
                pd.Dataframe:A dataframe of songs' information
                list: A list of labels predicted via kmeans clustering
    '''
    pred_labels=KMeans(n_clusters=n_cluster, n_init=10).fit_predict(df)
    summary_df=pd.DataFrame(df)
    summary_df['pred_clustering'] = pred_labels.tolist()
    print(f" Count by cluster")
    print(summary_df['pred_clustering'].value_counts())
    # print(summary_df.groupby(by='pred_clustering').agg(['count','mean']))
    print(f" Average values by variables and cluster")
    print(summary_df.groupby(by='pred_clustering').mean())
    pred_labels=summary_df.pop('pred_clustering')
    return summary_df, pred_labels.tolist()



def run_hdbscan(df:pd.Series, min_cluster_size:int):
    '''
            Peform HDBSCAN clustering

            Args in: df - full dataset of features to cluster.
                     min_cluster_size - minimum number of data points in a cluster
   
            Returns:
                   DataFrame: a dataframe with labels predicted via HDBSCAN 
                   list: a list of labels predicted via DBSCAN
    '''

    pred_labels=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(df)
    summary_df=pd.DataFrame(df)
    summary_df['pred_clustering'] = pred_labels.tolist() 
    print(f" Count by cluster")
    print(summary_df['pred_clustering'].value_counts())
    # print(summary_df.groupby(by='pred_clustering').agg(['count','mean']))
    print(f" Average values by variables and cluster")
    print(summary_df.groupby(by='pred_clustering').mean())
    pred_labels=summary_df.pop('pred_clustering')      

    return summary_df, pred_labels


def search_best_pca_n_components(df, range_n_components):
        '''
            Searches the best number of components for principal component analysis 
            Args in:
                    df-dataset of features to clusters
                    range_n_components :range for number of components to search
            Returns: None
                
        '''
        pca  = PCA(n_components=range_n_components)
        pca_data= pca.fit_transform(df)
        cum_var_expl = np.cumsum(pca.explained_variance_/np.sum(pca.explained_variance_))
        cum_var_expl
        plt.figure(figsize=(6,4))
        plt.plot(cum_var_expl)
        plt.xlabel('n_components')
        plt.ylabel('cumulative_explined_variance')
        plt.title("Amount of total variance included in the principal components")
        plt.show()
        plt.close()

def run_pca(df: pd.Series,n_components:int)->pd.Series:
        '''
            Peforms principal component analysis 
            Args in:
                    df-dataset of features to clusters
                    n_components :number of components
            Returns: 
                pd.Series: result of pca 
                
        '''
        pca=PCA(n_components=n_components)
        pca_data= pca.fit_transform(df)
        return pca_data


def clustering_visualisation(df:pd.Series,labels:pd.Series, title:str, algorithm):
        '''
            Visualizes dataset in 2D using umap. 
            Args in:
                    df-dataset of features to clusters
                    labels: labels from dataset (a column in dataset) that is used in order to compare with the labels predicted via kmeans clustering
                    n_cluster :number of clusters
                    algorithm: algorithm for visualisation
            Returns: 
                pd.Dataframe:A dataframe
                list: A list of labels predicted via kmeans clustering
        '''
        projections=algorithm(random_state=1).fit_transform(df)
        plt.scatter(projections[:,0],projections[:,1], c=labels)
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        plt.title(title)
        plt.show()
        plt.close()