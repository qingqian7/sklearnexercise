import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture

color_iter = itertools.cycle(['navy','c','cornflowerblue','gold','darkorange'])

def plot_result(X,Y_,means,covariances,index,title):
    splot = plt.subplot(2,1,index+1)
    for i,(mean,covar,color) in enumerate(zip(means,covariances,color_iter)):
        v,w = linalg.eigh(covar)
        v = 2* np.sqrt(2)*np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i,0],X[Y_ == i, 1],.8,color = color)
        angel = np.arctan(u[1]/u[0])
        angel = angel * 180 / np.pi
        ell = mpl.patches.Ellipse(mean,v[0],v[1],180+angel,color = color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.xlim(-9,5)
    plt.ylim(-3,6)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

# n_samples = 500
# np.random.seed(0)
# C = np.array([[0,-0.1],[1.7,0.4]])
# X = np.r_[np.dot(np.random.rand(n_samples,2),C),0.7*np.random.randn(n_samples,2)+np.array([-6,3])]

n_samples = 100
np.random.seed(0)
X = np.zeros((n_samples,2))
step = 4 * np.pi / n_samples
for i in range(X.shape[0]):
    x = i * step - 6
    X[i,0] = x + np.random.normal(0,0.1)
    X[i,1] = 3*(np.sin(x) + np.random.normal(0,0.2))
gmm = mixture.GaussianMixture(n_components=5,covariance_type='full').fit(X)
plot_result(X,gmm.predict(X),gmm.means_,gmm.covariances_,0,'Gaussian Mixture')
dpgmm = mixture.BayesianGaussianMixture(n_components=5,covariance_type='full').fit(X)
plot_result(X,dpgmm.predict(X),dpgmm.means_,dpgmm.covariances_,1,'Bayesian Gaussian Mixture with a Dirichlet process prior')
plt.show()
