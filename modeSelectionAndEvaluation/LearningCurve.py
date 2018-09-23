import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve,ShuffleSplit

def plot_learning_curve(estimator,title,x,y,ylim=None,cv = None,n_jobs=1,train_sizes=np.linspace(.1,1.0,5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("train examples")
    plt.ylabel("score")
    train_sizes,train_scores,test_scores = learning_curve(estimator,x,y,cv = cv,
                                                          n_jobs=n_jobs,train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis = 1)

    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean - train_scores_std,train_scores_mean + train_scores_std,alpha = 0.1,color='r')
    plt.fill_between(train_sizes,test_scores_mean - test_scores_std,test_scores_mean + test_scores_std,alpha=.1,color='g')

    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label='training score')
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='cross-validation score')

    plt.legend(loc='best')
    return plt


digits = load_digits()
x,y = digits.data,digits.target
cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
title = 'learning curve(naive bayes)'
estimator = GaussianNB()
plot_learning_curve(estimator,title,x,y,ylim=(0.7,1.01),cv = cv,n_jobs=1)

title = 'learning curves(svm with rbf kernel,$\gamma=0.001$)'
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator,title,x,y,ylim=(0.7,1.01),cv  =cv,n_jobs=1)
plt.show()


