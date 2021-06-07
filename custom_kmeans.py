#  =========================================================
#  HW 4: Unsupervised Learning, K-Means Clustering
#  CS 4824 / ECE 4484, Spring '21
#  Written by Matt Harrington, Haider Ali
#  =========================================================

"""
Adam Rankin
adamr14

"""


# Standard imports
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class CustomKMeans():

    # Initialize all attributes
    def __init__(self, k):
        self.k_ = k       # Number of clusters
        self.labels_ = 0  # Each sample's cluster label
        self.inertia_ = 0 # Sum of all samples' distances from their centroids

    # Find K cluster centers & label all samples
    def fit(self, data, plot_steps=False):
        # Fit the PCA module & Transform our data for later graphing
        self.pca = PCA(2).fit(data)
        self.data = pd.DataFrame(data)
        self.data_pca = pd.DataFrame(self.pca.transform(data))
        self.data_pca.columns = ['PC1', 'PC2']
        
        # Initialize variables
        self.iteration = 1
        n = data.shape[0]
        
        # Initialize centroids to random datapoints
        self.centroids = data.iloc[np.random.choice(range(n), self.k_, replace=False)].copy()
        self.centroids.index = np.arange(self.k_)
        
        #  =====================================================================
        #  ====================== TODO: IMPLEMENT KMEANS ======================= 
        
        #  while (not converged): # psuedocode - up to you to implement stopping criterion
        prev_inertia = -1
        while prev_inertia != self.inertia_:
            prev_inertia = self.inertia_
            self.assign_data() # Update
            self.update_centroids()     

        # show data & centroids at each iteration when testing performance
            if plot_steps:
                self.plot_state() 
                self.iteration += 1
        #  =====================================================================
            
        
        return self
    
    def assign_data(self):
        data = self.data.to_numpy()
        centroids = self.centroids.to_numpy()
        dist = np.array([]).reshape(data.shape[0],0)
        #
        # For each cluster calculate the distance between each point and the cluster
        #
        for k in range(self.k_):
            temp_dist = np.sum((data-centroids[k,:])**2, axis=1)
            dist = np.c_[dist, temp_dist]
            
        # assign the minimum distance to the cluster
        self.labels_ = np.argmin(dist, axis=1)
                
    def update_centroids(self):
        grouped_records = {}
        data = self.data.to_numpy()
        # initialize
        for k in range(self.k_):
            grouped_records[k]=np.array([]).reshape(17,0)
            
        # add records to approriate group
        for i in range(data.shape[0]):
            grouped_records[self.labels_[i]]=np.c_[grouped_records[self.labels_[i]], data[i]]
            
        # Build new array of centroids    
        centroids = np.array([]).reshape(17,0)
        for k in range(self.k_):
            centroids=np.c_[centroids, np.mean(grouped_records[k].T, axis=0)]
        self.centroids = pd.DataFrame(data=centroids.T, index=np.arange(self.k_))
        
        # calculate inertia
        inertia = 0
        for k in range(self.k_):
            inertia += np.sum((grouped_records[k].T-centroids.T[k,:])**2)
        self.inertia_ = inertia
        
    # Plot projection of data and centroids in 2D
    def plot_state(self):
        # Project the centroids along the principal components
        centroid_pca = self.pca.transform(self.centroids)
        
        # Draw the plot
        plt.figure(figsize=(8,8))
        plt.scatter(self.data_pca['PC1'], self.data_pca['PC2'], c=self.labels_)
        plt.scatter(centroid_pca[0], centroid_pca[1], marker = '*', s=1000)
        plt.title("Clusters and Centroids After step {}".format(self.iteration))
        plt.show()