import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA,KernelPCA
from sklearn.datasets import make_circles
np.random.seed(0)
x, y = make_circles(n_samples=400,factor=.3,noise=.05)
kpca = KernelPCA(kernel='rbf',fit_inverse_transform=True,gamma=10)
x_kpca = kpca.fit_transform(x)
x_back = kpca.inverse_transform(x_kpca)
pca = PCA()
x_pca = pca.fit_transform(x)

plt.figure()
plt.subplot(2,2,1,aspect='equal')
plt.title('original space')
reds = y == 0
blues = y == 1
plt.scatter(x[reds,0],x[reds,1],c="red",s=20,edgecolor='k')
plt.scatter(x[blues,0],x[blues,1],c="blue",s=20,edgecolor='k')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

#################################
# np.random.seed(0)

# X, y = make_circles(n_samples=400, factor=.3, noise=.05)
#
# kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
# X_kpca = kpca.fit_transform(X)
# X_back = kpca.inverse_transform(X_kpca)
# pca = PCA()
# X_pca = pca.fit_transform(X)
#
# # Plot results
#
# plt.figure()
# plt.subplot(2, 2, 1, aspect='equal')
# plt.title("Original space")
# reds = y == 0
# blues = y == 1
#
# plt.scatter(X[reds, 0], X[reds, 1], c="red",
#             s=20, edgecolor='k')
# plt.scatter(X[blues, 0], X[blues, 1], c="blue",
#             s=20, edgecolor='k')
# plt.xlabel("$x_1$")
# plt.ylabel("$x_2$")
# plt.show()
x1, x2 = np.meshgrid(np.linspace(-1.5,1.5,50),np.linspace(-1.5,1.5,50))
x_grid = np.array([np.ravel(x1),np.ravel(x2)]).T
z_grid = kpca.transform(x_grid)[:,0].reshape(x1.shape)
plt.contour(x1,x2,z_grid,color="grey",linewidth=1,origin='lower')

plt.subplot(2,2,2,aspect="equal")
plt.scatter(x_pca[reds,0],x_pca[reds,1],c='red',s=20,edgecolors='k')
plt.scatter(x_pca[blues,0],x_pca[blues,1],c='blue',s=20,edgecolors='k')
plt.title('projection by pca')

plt.subplot(2,2,3,aspect='equal')
plt.scatter(x_kpca[reds,0],x_kpca[reds,1],c='red',s=20,edgecolors='k')
plt.scatter(x_kpca[blues,0],x_kpca[blues,1],c='blue',s=20,edgecolors='k')
plt.subplot(2, 2, 4, aspect='equal')
plt.scatter(x_back[reds, 0], x_back[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(x_back[blues, 0], x_back[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.show()
