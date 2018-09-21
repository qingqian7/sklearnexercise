import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm
from sklearn.feature_selection import SelectPercentile,f_classif

iris = datasets.load_iris()
E = np.random.uniform(0,0.1,size=(len(iris.data),20))
x = np.hstack((iris.data,E))
y = iris.target
plt.figure(1)
plt.clf()
x_indices = np.arange(x.shape[-1])

selector = SelectPercentile(f_classif,percentile=10)
selector.fit(x,y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(x_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
        edgecolor='black')

clf = svm.SVC(kernel='linear')
clf.fit(x,y)
svm_weights = (clf.coef_ ** 2).sum(axis = 0)
svm_weights /= svm_weights.max()
plt.bar(x_indices - .25, svm_weights, width=.2, label='SVM weight',
        color='navy', edgecolor='black')
# plt.show()
clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(x),y)

svm_weight_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weight_selected /= svm_weight_selected.max()
plt.bar(x_indices[selector.get_support()] - .05, svm_weight_selected,
        width=.2, label='SVM weights after selection', color='c',
        edgecolor='black')
plt.title("comparing feature selection")
plt.xlabel("Feature number")
plt.yticks()
plt.axis("tight")
plt.legend(loc='upper right')
plt.show()