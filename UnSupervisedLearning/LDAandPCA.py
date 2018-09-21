import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
x_r = pca.fit(x).transform(x)

lda = LinearDiscriminantAnalysis(n_components=2)
x_r2 = lda.fit(x,y).transform(x)
print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
plt.figure()
colors=['navy','turquoise','darkorange']
# print(x_r)
lw = 2
for color, i, target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(x_r[y == i,0],x_r[y == i,1],color=color,alpha=0.8,lw = lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.figure()
for color, i, target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(x_r2[y == i,0],x_r2[y == i,1],color = color,alpha=0.8,label=target_name)
plt.legend(loc='best',shadow=False,scatterpoints=1)
plt.title('LDA of iris data')
plt.show()