import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture

color_iter = itertools.cycle(['navy','c','cornflowerblue','gold','darkorange'])

def plot_results(X,Y,means,covariances,index,title):
    splot = plt.subplot(5,1,index + 1)
    for i,(mean,covar,color) in enumerate(zip(means,covariances,color_iter)):
        v,w = linalg.eigh(covar)
        v = 2 * np.sqrt(2) * np.sqrt(v)
        u = w[0] /linalg.norm(w[0])
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i,0],X[Y == i,1],0.8,color = color)

        #plot an ellipse to show the gaussian component
        angle = np.arctan([1]/u[0])
        angle = angle * 180 /np.pi
        ell = mpl.patches.Ellipse(mean,v[0],v[1],180 + angle,color = color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.xlim(-6,4 * np.pi - 6)
    plt.ylim(-5,5)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())

def plot_samples(X,Y)：