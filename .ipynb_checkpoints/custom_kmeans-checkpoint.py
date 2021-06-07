#  =========================================================
#  HW 4: Unsupervised Learning, K-Means Clustering
#  CS 4824 / ECE 4484, Spring '21
#  Written by Matt Harrington, Haider Ali
#  =========================================================

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
        
        self.labels_ = [random.randint(0,self.k_-1) for i in range(n)] # Update
        self.inertia_ = np.sum(np.arange(n))                           # Update
        
        # show data & centroids at each iteration when testing performance
        if plot_steps:
            self.plot_state() 
            self.iteration += 1
        #  =====================================================================
            
        
        return self
    
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