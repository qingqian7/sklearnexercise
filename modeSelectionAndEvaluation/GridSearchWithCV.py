from __future__ import print_function
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

digits = datasets.load_digits()
n_samples = len(digits.images)
x = digits.images.reshape((n_samples,-1))
y = digits.target
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=0)
tuned_parameters = [{'kernel':['rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]},{'kernel':['linear'],'C':[1,10,100,1000]}]
scores = ['precision','recall']
for score in scores:
    print("# tuning hyper-parameters for %s "% score)
    print()
    clf = GridSearchCV(SVC(),tuned_parameters,cv = 5,scoring='%s_macro' % score)
    clf.fit(x_train,y_train)
    print("best parameters set found on development set")
    print()
    print(clf.best_params_)
    print()
    print('grid scores on development set:')
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean,std,params in zip(means,stds,clf.cv_results_['params']):
        print("%0.3f (+/-%0.3f) for %r" % (mean,std *2,params))
    print()
    print("detailed classification report ")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true,y_pred))
    print()