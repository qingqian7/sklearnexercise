import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

x,y = make_classification(n_samples=1000,n_features=25,n_informative=4,n_redundant=2,n_repeated=0,n_classes=8,
                          n_clusters_per_class=1,random_state=0)
svc = svm.SVC(kernel='linear')
rfecv = RFECV(estimator=svc,step=1,cv=StratifiedKFold(2),scoring='accuracy')
rfecv.fit(x,y)
print('optional number of features %d : '% rfecv.n_features_)
print('ranking of features: %s' % rfecv.ranking_)
plt.figure()
plt.xlabel('number of features selected ')
plt.ylabel('scores of cross validation (nb of correct classfication)')
plt.plot(range(1,len(rfecv.grid_scores_)+1),rfecv.grid_scores_)
plt.show()