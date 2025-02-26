'''kmeans.py
Performs K-Means clustering
Henry Landay
CS 251/2: Data Analysis and Visualization
Spring 2024
'''
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            # data: ndarray. shape=(num_samps, num_features)
            self.data = data.copy()
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return np.copy(self.data)

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.sqrt(np.sum((pt_1-pt_2)**2))

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.sqrt(np.sum((pt-centroids)**2, axis=1))

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        indices = np.random.choice(self.data[:, 0], k)
        indices = indices.astype(int)
        return self.data[indices]

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, p=2):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        if self.num_samps < k:
            raise RuntimeError('Cannot compute kmeans with #data samples < k!')
        if k < 1:
            raise RuntimeError('Cannot compute kmeans with k < 1!')

        self.k = k
        self.centroids = self.initialize(k)
        self.data_centroid_labels = self.update_labels(self.centroids)
        self.centroids, centroid_dif = self.update_centroids(
            self.k, self.data_centroid_labels, self.centroids)

        i = 0
        while i < max_iter and np.sum(np.abs(centroid_dif)) > tol:
            i += 1
            self.data_centroid_labels = self.update_labels(self.centroids)
            self.centroids, centroid_dif = self.update_centroids(
                k, self.data_centroid_labels, self.centroids)
            self.inertia = self.compute_inertia()

        # print total iterations
        # print(i)

        return self.inertia

    def cluster_batch(self, k=2, n_iter=1, verbose=False, p=2):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        centroids = []
        labels = []
        inertias = np.array(np.ones(n_iter))
        for j in range(n_iter):
            inertia = self.cluster(k)
            centroids.append(self.centroids)
            labels.append(self.data_centroid_labels)
            inertias[j] = inertia
        index = np.where(inertias == np.min(inertias))[0][0]
        self.inertia = inertias[index]
        self.centroids = centroids[index]
        self.data_centroid_labels = labels[index]
        
    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        labels = np.ones(self.num_samps)
        for i in range(len(self.data)):
            dist = self.dist_pt_to_centroids(self.data[i, :], centroids)
            min_index = np.argmin(dist)
            labels[i] = min_index
        labels = labels.astype(int)
        return labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        # new_centroids = np.ones((k, self.num_features))
        # centroids_dif = np.ones((k, self.num_features))
        # for i in range(k):
        #     # checking to see if the sum of the truth values sum to zero
        #     if np.sum(data_centroid_labels == i) == 0:
        #         index = np.random.choice(np.arange(self.num_samps-1))
        #         new_centroids[i] = self.data[index]
        #     else:
        #         means = np.mean(self.data[data_centroid_labels == i], axis=0)
        #         new_centroids[i, :] = means
        # centroids_dif = new_centroids - prev_centroids
        # return new_centroids, centroids_dif

        new_centroids = np.ones((k, self.num_features))
        centroid_diff = np.ones((k, self.num_features))
        for i in range(k) :
            if np.sum(data_centroid_labels == i) == 0 :
                # new_centroids[i] = self.data[np.random.choice(self.data[:,0]).astype(int)]
                idx = np.random.choice(np.arange(self.num_samps-1))
                new_centroids[i] = self.data[idx]
            
            # average all x and y values to get the point for the new centroid
            else :
                new_centroids[i,:] = np.mean(self.data[data_centroid_labels == i], axis=0)
                
        centroid_diff = new_centroids - prev_centroids
        return new_centroids, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        totalDist = 0
        for i in range(self.num_samps):
            totalDist += (self.data[i][0]-self.centroids[self.data_centroid_labels[i]][0])**2 + (
                self.data[i][1]-self.centroids[self.data_centroid_labels[i]][1])**2
        inertia = totalDist/len(self.data)
        return inertia

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). To make sure you change your colors to be clearly differentiable,
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        Each string in the `colors` list that starts with # is the hexadecimal representation of a color (blue, red, etc.)
        that can be passed into the color `c` keyword argument of plt.plot or plt.scatter.
            Pick one of the palettes with a generous number of colors so that you don't run out if k is large (e.g. >6).
        '''
        # 10 colors
        colors = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
        data_with_labels = np.hstack((self.data, self.data_centroid_labels[:, np.newaxis]))
        for i in range(self.k):
            temp = data_with_labels[data_with_labels[:,2] == i]
            plt.scatter(temp[:,0], temp[:,1], color=colors[i], label=f'Cluster {i}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='*', label='Centroid')
        plt.title('Area vs. Asymmetry')
        plt.xlabel('Area')
        plt.ylabel('Asymmetry')
        plt.legend(fontsize=13)

    def elbow_plot(self, max_k, n_iter=1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        n_iter: int. Create and graph n elbow plots

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''

        x_line = np.arange(1, max_k+1)
        inertia = np.ones(max_k)
        
        for i in range(max_k):
            self.cluster_batch(k=i+1, n_iter=n_iter)
            inertia[i] = self.inertia
        plt.plot(x_line, inertia)
        plt.xticks(x_line)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.show()

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        for i in range(len(self.data)) :
            dist = self.dist_pt_to_centroids(self.data[i,:],self.centroids)
            minIDX = np.argmin(dist)
            self.data[i,:] = self.centroids[minIDX]

