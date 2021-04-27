seed = 5953
import numpy as np
from scipy.stats import norm, multivariate_normal
import random
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
colors = ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige", "brown", "gray", "cyan", "magenta"]
style.use('fivethirtyeight')

np.random.seed(seed)
random.seed(seed)


def gen_data(n_samples):
    # generating some random sample data
    # a mixture of ellipsoidal and appx circular shape

    # data 1
    data1 = np.random.randn(n_samples, 2) + np.array([8, 10])
    
    # data 2
    C = np.array([[0., 0.7], [1.5, 0.9]])
    data2 = np.dot(np.random.randn(n_samples, 2), C) + np.array([4, 5])

    # data 3
    C = np.array([[1., -0.3], [0.5, 1.4]])
    data3 = np.dot(np.random.randn(n_samples, 2) + np.array([0, 7]), C)

    # concatenate datasets into the final data set
    data = [data1, data2, data3]
    cols = ['r','b','g']
    X = np.vstack([data1, data2, data3])
    X = np.vstack(data)

    return X, data, cols

def plot_data(X, data, cols, filename='output/data_plot_labels.png'):
    plt.style.use('seaborn')
    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], 2.5) 
    fig.savefig('output/data_plot.png')

    fig2, ax2 = plt.subplots()    
    for i in range(len(data)):
        ax2.scatter(data[i][:, 0], data[i][:, 1], 0.80, facecolor=cols[i]) 

    ax.grid(True)
    fig2.savefig(filename)
   
    return 1


class GaussianMixtureModelmd():
    """
    A Gaussian mixture model for 2d data
    """

    def __init__(self, x, num_components):
        """
        Initialize a Gaussian mixture model.

        params:
        x = 1d numpy array data
        num_components = int
        """
        self.x = x
        self.shape = x.shape
        self.num_components = num_components
        self.means = np.array(x[np.random.choice(x.shape[0], num_components, replace=False)])      
        self.covs = np.full((num_components, x.shape[1], x.shape[1]), np.cov(x, rowvar = False))
        self.prior_mixture = np.ones(num_components)/num_components
        self.probs = np.zeros([len(x), num_components])
        self.responsibilities = np.zeros([len(x), num_components])
        self.log_likelihood = 0
        self.log_likelihood_trace = []
        self.converged = False

    def train(self, iters):
        for j in range(iters):
            print(f'iter={j}  mean={self.means}')
            self.e_step()
            self.m_step()
    
    def e_step(self):
        """
        expectation step: computing responsibilities of each data cluster 
        """
        for i in range(self.num_components):
            self.probs[:,i] = np.multiply(self.prior_mixture[i], multivariate_normal(self.means[i], self.covs[i]).pdf(self.x))
            
        self.log_likelihood = np.sum(np.log(np.sum(self.probs, axis = 1)))
        self.responsibilities = self.probs/self.probs.sum(axis=1)[:,None]
        
    def m_step(self):
        """
        maximization step: updating means and covariances
        """
        for i in range(self.num_components):
            self.prior_mixture[i] = np.divide(np.sum(self.responsibilities[:,i], axis=0), len(self.x))
            self.means[i] = np.sum(self.responsibilities[:,i].reshape(-1, 1)*self.x)/np.sum(self.responsibilities[:,i])
            diff = (self.x - self.means[i]).T
            weighted_sum = np.dot(self.responsibilities[:, i] * diff, diff.T)
            self.covs[i] = np.divide(weighted_sum, np.sum(self.responsibilities[:,i], axis=0))
            
    def get_means(self):
        return self.means
        
    def get_stdevs(self):
        return self.covs
        
    def get_responsibilities(self):
        return self.responsibilities


if __name__ == '__main__':

    # generate random data clusters
    n_samples = 250
    X, data, cols = gen_data(n_samples)
    plot_data(X, data, cols, 'output/data_plot_labels.png')

    # running the gmm
    gmm1d = GaussianMixtureModelmd(X, 3)
    gmm1d.train(10)

    