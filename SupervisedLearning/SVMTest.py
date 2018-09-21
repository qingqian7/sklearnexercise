import numpy as np
from sklearn import datasets,svm
import matplotlib.pyplot as plt

def make_meshgrid(x,y,h = .02):
    x_min, x_max = x.min() -1, x.max() +1
    y_min, y_max = y.min() -1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    return xx,yy
def plot_contours(ax, clf, xx, yy, **params):
    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    out = ax.contourf(xx, yy, z, **params)
    return out

#load data
iris = datasets.load_iris()
x = iris.data[:,:2]
y = iris.target
C = 1.0
models = (svm.SVC(kernel='linear',C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf',gamma=0.7,C=C),
          svm.SVC(kernel='poly',degree=3,C=C))
models = (clf.fit(x,y) for clf in models)
titles = ('SVC with linear kernel','LinearSVC (linear kernel)','SVC with rbf kernel','SVC with ploy kernel(degree=3)')
fig, sub = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.4,hspace=0.4)
x0,x1 = x[:,0],x[:,1]
xx, yy = make_meshgrid(x0,x1)
for clf ,title, ax in zip(models,titles,sub.flatten()):
    plot_contours(ax,clf,xx,yy,cmap=plt.cm.coolwarm,alpha=0.8)
    ax.scatter(x0, x1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.show()