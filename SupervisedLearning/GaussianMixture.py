import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

n_samples = 300
#generate random samples ,two components
np.random.seed(0)
#generate spherical data centered (20,20)
shifted_gaussian = np.random.randn(n_samples,2) + np.array([20,20])

#generate zero centered streched Gaussian data
C = np.array([[0.,-0.7],[3.5,.7]])
streched_gaussian = np.dot(np.random.rand(n_samples,2),C)

#concatenate the two datesets into the final training set
x_train = np.vstack([shifted_gaussian,streched_gaussian])

#fit a Gaussian Mixture model with two components
clf = mixture.GaussianMixture(n_components=2,covariance_type='full')
clf.fit(x_train)

#display the predicted scores by the model as a contour plot
x = np.linspace(-20,30)
y = np.linspace(-20,40)
X, Y = np.meshgrid(x,y)
xx = np.array([X.ravel(),Y.ravel()]).T
z = -clf.score_samples(xx)
z = z.reshape(X.shape)
CS = plt.contour(X,Y,z,norm=LogNorm(vmin=1.0,vmax=1000.0),levels=np.logspace(0,3,10))
CB = plt.colorbar(CS,shrink=0.8,extend='both')
plt.scatter(x_train[:,0],x_train[:,1],.8)
plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()
