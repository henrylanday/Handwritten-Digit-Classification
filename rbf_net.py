'''rbf_net.py
Radial Basis Function Neural Network
YOUR NAME HERE
CS 252: Mathematical Data Analysis and Visualization
Spring 2024
'''
import numpy as np

from kmeans import KMeans
from classifier import Classifier


class RBF_Net(Classifier):
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Call the superclass constructor
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        super().__init__(num_classes=num_classes)
        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None

        self.k = num_hidden_units
        self.num_classes = num_classes

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes

    def get_wts(self):
        '''Returns the hidden-output layer weights and bias

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(num_hidden_units+1, num_classes).
        '''
        return self.wts

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_classes

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''
        k = centroids.shape[0]
        average_distances = []
        for i in range(k):
            clusters_data = data[cluster_assignments == i]
            centroid = centroids[i]
            distances = np.linalg.norm(clusters_data - centroid, axis=1)
            average_distances.append(np.mean(distances))
        return np.array(average_distances)

    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        kmeans = KMeans(data)
        kmeans.cluster_batch(k=self.k, n_iter=5)
        self.prototypes = kmeans.get_centroids()
        self.sigmas = self.avg_cluster_dist(
            data, kmeans.get_centroids(), kmeans.get_data_centroid_labels(), kmeans)

    def pseudo_inverse(self, A):
        '''Uses the SVD to compute the pseudo-inverse of the data matrix `A`

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.

        Returns
        -----------
        ndarray. shape=(num_features, num_data_samps). The pseudoinverse of `A`.

        NOTE:
        - You CANNOT use np.linalg.pinv here!! Implement it yourself with SVD :)
        - Skip this until we cover the topic in lecture
        '''
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        S = 1/S
        S = np.diag(S)
        return (U @ S @ Vt).T

    def linear_regression(self, A, y):
        '''Performs linear regression using the SVD-based solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE:
        - Remember to handle the intercept
        - You should use your own SVD-based solver here, but if you get here before we cover this in lecture, use
        scipy.linalg.lstsq for now.
        '''
        A_new = np.hstack((np.ones((A.shape[0], 1)), A))
        A_plus = self.pseudo_inverse(A_new)
        c = A_plus @ y
        return c

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        # '''

        h = np.zeros([data.shape[0], self.k])
        for i in range(len(data)):
            for j in range(self.k):
                h[i][j] = np.exp(-1 * (np.linalg.norm(data[i, :] - self.prototypes[j, :])
                                 ** 2) / ((2 * self.sigmas[j] ** 2) + (1e-08)))
        return h

    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        ones = np.ones((hidden_acts.shape[0], 1))
        H = np.hstack((ones, hidden_acts))
        # print(H)
        # print(self.wts)
        output_act = np.dot(H, self.wts)
        return output_act

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        self.initialize(data)
        hidden_act = self.hidden_act(data)
        self.wts = np.ones((self.k+1, self.num_classes))
        for i in range(self.num_classes):
            vector = np.zeros((y.shape[0]))
            idx = np.where(y == i)
            vector[idx] = 1
            self.wts[:, i] = self.linear_regression(hidden_act, vector)

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''
        hidden = self.hidden_act(data)
        output = self.output_act(hidden)
        return np.argmax(output, axis=1)
