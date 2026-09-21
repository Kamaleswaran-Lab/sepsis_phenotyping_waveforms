from __future__ import absolute_import
import scikitplot as skplt
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Ref: https://github.com/burtonrj/consensusclustering
from consensusclustering import ConsensusClustering
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import DistanceMetric

def tc_consensus_clustering(data, k_opt=5, show='False'):
    '''
    Input args:
        data: 2-D input data array with scaling is preferred
        show: binary plotting flag (True, False)
    Output args: 
        consensus_label
    '''
    clustering_obj = AgglomerativeClustering(metric='euclidean', linkage='average') 
    cc = ConsensusClustering(
        clustering_obj=clustering_obj,
        min_clusters=2,
        max_clusters=9,
        n_resamples=100, #Number of resamples to perform.
        resample_frac=0.8, #Fraction of rows to resample
        k_param='n_clusters'
    )
    cc.fit(data, progress_bar=True, n_jobs=10)

    if show:
        _, ax = plt.subplots(figsize=(3.5, 3.5))
        cc.plot_cdf(ax=ax)
        ax.legend(bbox_to_anchor=(1, 1))
        plt.show()
        
        _, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        cc.plot_auc_cdf(ax=ax[0], include_knee=False)
        cc.plot_change_area_under_cdf(ax=ax[1])
        plt.show()
    
        _, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        for i, ax in enumerate(axes):
            cc.plot_hist(i + 3, ax=ax)
            ax.set_title('K = '+str(i+3))
        plt.show()

        grid = cc.plot_clustermap(
            k=3, 
            figsize=(5, 5), 
            dendrogram_ratio=0.1, 
            xticklabels=False,
            yticklabels=False
        )
        grid.cax.set_visible(False)

        del grid

        grid = cc.plot_clustermap(
            k=4, 
            figsize=(5, 5), 
            dendrogram_ratio=0.1, 
            xticklabels=False,
            yticklabels=False)
        grid.cax.set_visible(False)

        del grid

        grid = cc.plot_clustermap(
            k=5, 
            figsize=(5, 5), 
            dendrogram_ratio=0.1, 
            xticklabels=False,
            yticklabels=False)
        grid.cax.set_visible(False)

    print(f"Best k from change in AUC: {cc.best_k('change_in_auc')}, and from AUC knee point: {cc.best_k('knee')}")
    con_clust_labels = get_cluster_labels(consensus_matrix_3d=cc.consensus_matrices_, k_min=2, k_max=9, k_opt=k_opt)
    return con_clust_labels

    
'''
Consensus Cluster labels:

'''

def get_cluster_labels(consensus_matrix_3d, k_min, k_max, k_opt):
    
    k = np.arange(k_min, k_max+1)
    id_opt = np.where(k==k_opt)[0][0]
    consensus_matrix = consensus_matrix_3d[id_opt]
    
    # Convert the consensus matrix to a distance matrix
    dist_matrix = 1 - consensus_matrix
    
    # Perform hierarchical clustering
    Z = linkage(pairwise_distances(dist_matrix), method='average')
    
    # Extract cluster labels for the specified number of clusters
    labels = fcluster(Z, k_opt, criterion='maxclust')-1
    
    return labels




def metric(x, pred_y):
    skplt.metrics.plot_silhouette(x, pred_y)
    plt.show()

    ss = np.round(silhouette_score(x, pred_y, metric='euclidean'),3)
    db = np.round(sklearn.metrics.davies_bouldin_score(x, pred_y),3)
    ch = np.round(sklearn.metrics.calinski_harabasz_score(x, pred_y),3)
    
    print('SS, DB, CH: '+str(ss)+', '+ str(db) +', '+ str(ch))
    
    
    
    
    
    