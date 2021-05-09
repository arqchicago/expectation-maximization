import numpy as np
from scipy.stats import norm
from random import random
np.random.seed(0)


class GaussianMixtureModel1d():
    """
    A Gaussian mixture model for 1d data
    """

    def __init__(self, x, num_components):
        """
        Initialize a Gaussian mixture model.

        params:
        x = 1d numpy array data
        num_components = int
        """
        self.x = x
        self.num_components = num_components
        self.means = np.array(np.random.choice(x, num_components))
        self.stdevs = np.ones(num_components)
        self.prior_mixture = np.ones(num_components)/num_components
        self.probs = np.zeros([len(x), num_components])
        self.responsibilities = np.zeros([len(x), num_components])

    def train(self, iters):
        for j in range(iters):
            self.e_step()
            self.m_step()
    
    def e_step(self):
        """
        expectation step: computing responsibilities of each data cluster 
        """
        for i in range(self.num_components):
            self.probs[:,i] = np.multiply(self.prior_mixture[i]*1/np.sqrt(2*np.pi*(self.stdevs[i])**2), np.exp(np.multiply(-0.50, np.square(np.divide(self.x-self.means[i], self.stdevs[i])))))
        
        self.responsibilities = self.probs/self.probs.sum(axis=1)[:,None]
    
    def m_step(self):
        """
        maximization step: updating means and standard deviations
        """
        for i in range(self.num_components):
            self.prior_mixture[i] = np.divide(np.sum(self.responsibilities[:,i], axis=0), len(x))
            self.means[i] = np.sum(self.responsibilities[:,i]*x)/np.sum(self.responsibilities[:,i])
            self.stdevs[i] = np.sqrt(np.sum(self.responsibilities[:,i]*np.square(self.x-self.means[i]))/np.sum(self.responsibilities[:,i]))

    def get_means(self):
        return self.means
        
    def get_stdevs(self):
        return self.stdevs


if __name__ == '__main__':

    # generate random data clusters
    X_orig = np.linspace(-5,5,num=20)
    X0 = X_orig*np.random.rand(len(X_orig))+15 # create data cluster 1
    X1 = X_orig*np.random.rand(len(X_orig))-15 # create data cluster 2
    X2 = X_orig*np.random.rand(len(X_orig)) # create data cluster 3
    x = np.stack((X0,X1,X2)).flatten() # combine the clusters to get the random datapoints from above
    
    # running the gmm
    gmm1d = GaussianMixtureModel1d(x, 3)
    gmm1d.train(15)
    means, std_devs = gmm1d.get_means(), gmm1d.get_stdevs()
    cluster_pars = zip(np.round(means, 4), np.round(std_devs, 4))
    print(f'gaussian clusters (mean, variance):  {list(cluster_pars)}')
    