seed = 5941
import numpy as np
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

np.random.seed(seed)


def draw_from_normal(mean, std, n):
    return np.random.normal(mean, std, size=n)


if __name__ == '__main__':

   
    #create clusters
    x0 = draw_from_normal(10,1.8,10)
    x1 = draw_from_normal(28,3.9,10)
    x2 = draw_from_normal(50,3.3,10)
    x = np.stack((x0,x1,x2)).flatten()
    
    # getting a visual
    x_list = [x0, x1, x2]
    y_list = [[0 for i in x_i] for x_i in x_list]

    print(x_list)
    print(y_list)

    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    
    for i in range(len(x_list)):
        ax.scatter(x_list[i], y_list[i])

    ax.set_yticks([0,0.25, 0.5, 0.75, 1])
    ax.grid(True)
    fig.savefig('output/feature_importance_plot.png')

    