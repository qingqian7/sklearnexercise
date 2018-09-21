import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

x,y = make_blobs(n_samples=40,centers=2,random_state=6)
clf = svm.SVC(kernel='linear',C=1000)
clf.fit(x,y)
plt.scatter(x[:,0],x[:,1],c=y)