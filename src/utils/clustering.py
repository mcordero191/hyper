'''
Created on 26 Apr 2024

@author: mcordero
'''

import numpy as np
# from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
    
def hierarchical_cluster(X, cmap='nipy_spectral', alpha=0.2):
    '''
    X[npoints, nfeatures]    :    Dataset with npoints and nfeatures
    '''
    # Create dataset with 20 features and 200 data points
    # np.random.seed(0)
    # X = np.random.rand(200, 20)
    
    xmin = np.min(X, axis=0)
    xmax = np.max(X, axis=0)
    xmean = np.mean(X, axis=0)
    
    X = (X - xmean[None,:])/(xmax-xmin)[None,:]
    
    npoints = len(X[:,0])
    
    print("Permorfing clustering ...")
    # Perform hierarchical clustering with complete linkage
    # linkage_matrix = linkage(X, method='complete', metric='euclidean')
    
    # Plot the dendrogram
    # plt.figure(figsize=(12, 6))
    # dendrogram(linkage_matrix, truncate_mode='lastp', p=20, leaf_rotation=90., leaf_font_size=8., show_contracted=True)
    # plt.title('Hierarchical Clustering Dendrogram with Complete Linkage')
    # plt.xlabel('Data Points')
    # plt.ylabel('Distance')
    # plt.show()

    # Specify the desired number of clusters
    # nlinks = len(np.unique(X[:,2]))
    
    n_clusters = 30
    
    # # Create an Agglomerative Clustering model
    # model = AgglomerativeClustering(n_clusters=n_clusters)
    # # Fit the model to the data and get cluster labels
    # cluster_labels = model.fit_predict(X)
    
    # define the model
    # model = DBSCAN(eps=0.30, min_samples=9)
    # fit model and predict clusters
    # cluster_labels = model.fit_predict(X)
    
    
    model = GaussianMixture(n_components=n_clusters)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    cluster_labels = model.predict(X)    

    min_idx = np.argmin(X[:,0])
    max_idx = np.argmax(X[:,0])
    
    min_label = model.predict(X[min_idx:min_idx+1,:])
    max_label = model.predict(X[max_idx:max_idx+1,:])
    
    valid = (cluster_labels != min_label) & (cluster_labels != max_label)
    
    if np.count_nonzero(~valid) > 0.05*npoints:
        valid[:] = True
    
    X_invalid = X[~valid,:]
    clabels_invalid = cluster_labels[~valid]
    
    plt.figure(figsize=(8, 4))
    
    # plt.subplot(121)
    # Plot the data with cluster labels
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap=cmap, alpha=alpha)
    
    # X = X[valid,:]
    # plt.scatter(X[:, 0], X[:, 1], marker='x', cmap=cmap, alpha=alpha)
    
    plt.title('Total counts: %d' %len(X[:,0]))
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.colorbar(label='Cluster')
    
    # plt.subplot(222)
    # # Plot the data with cluster labels
    # plt.scatter(X[:, 0], X[:, 2], c=cluster_labels, cmap=cmap, alpha=alpha)
    # # plt.title(f'Agglomerative Clustering with {n_clusters} Clusters')
    # plt.xlabel('Feature 0')
    # plt.ylabel('Feature 2')
    # plt.colorbar(label='Cluster')
    
    #
    # plt.subplot(122)
    # # Plot the data with cluster labels
    # plt.scatter(X_invalid[:, 0], X_invalid[:, 1], c=clabels_invalid, cmap=cmap, alpha=alpha)
    #
    # # X = X[valid,:]
    # # plt.scatter(X[:, 0], X[:, 1], marker='x', cmap=cmap, alpha=alpha)
    #
    # plt.title('Total counts: %d' %len(X_invalid[:,0]))
    # plt.xlabel('Feature 0')
    # plt.ylabel('Feature 1')
    # plt.colorbar(label='Cluster')
    
    # plt.subplot(224)
    # # Plot the data with cluster labels
    # plt.scatter(X_valid[:, 0], X_valid[:, 2], c=clabels_valid, cmap=cmap, alpha=alpha)
    # # plt.title(f'Agglomerative Clustering with {n_clusters} Clusters')
    # plt.xlabel('Feature 0')
    # plt.ylabel('Feature 2')
    # plt.colorbar(label='Cluster')
    
    
    plt.tight_layout()
    plt.show()
    
    return(valid)

if __name__ == '__main__':
    pass