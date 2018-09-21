from sklearn import svm,datasets
clf = svm.SVC()
iris = datasets.load_iris()
x, y = iris.data, iris.target
clf.fit(x,y)

# import pickle
# s = pickle.dumps(clf)
# clfs = pickle.loads(s)
# print(clfs.predict(x[0:1]))
from sklearn.externals import joblib
joblib.dump(clf,'clf.pkl')
clf2 = joblib.load('clf.pkl')
print(clf2.predict(x[0:1]))
