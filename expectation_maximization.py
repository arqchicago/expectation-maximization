seed = 5941
import numpy as np
from scipy.stats import norm
import random
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

np.random.seed(seed)



if __name__ == '__main__':

    # generating some random sample data
    # a mixture of ellipsoidal and appx circular shape
    
    n_samples = 250
    
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

    plt.style.use('seaborn')
    fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], 2.5) 
    fig.savefig('output/data_plot.png')

    fig2, ax2 = plt.subplots()    
    for i in range(len(data)):
        ax2.scatter(data[i][:, 0], data[i][:, 1], 0.80, facecolor=cols[i]) 

    ax.grid(True)
    fig2.savefig('output/data_plot_labels.png')

    