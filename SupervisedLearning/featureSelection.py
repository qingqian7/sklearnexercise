# # from sklearn.feature_selection import VarianceThreshold
# # x = [[0,0,1],[0,1,0],[1,0,0],[0,1,1],[0,1,0],[0,1,1]]
# # sel = VarianceThreshold(threshold=(.8*(1-.8)))
# # y = sel.fit_transform(x)
# # print(y)
#
# # SelectKBest  selection
# # from sklearn.datasets import load_iris
# # from sklearn.feature_selection import SelectKBest,chi2
# # iris = load_iris()
# # x,y = iris.data, iris.target
# # print(x.shape)
# # x_new = SelectKBest(chi2,k=3).fit_transform(x,y)
# # print(x_new.shape)



# # Comparison of F_test and mutual information
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.feature_selection import f_regression,mutual_info_regression
#
# np.random.seed(0)
# x = np.random.rand(1000,3)
# y = x[:,0] + np.sin(6 * np.pi * x[:,1]) + 0.1 * np.random.randn(1000)
#
# f_test,_ = f_regression(x,y)
# print(f_test)
# f_test /= np.max(f_test)
# print("########################")
# print(f_test)
# mi = mutual_info_regression(x,y)
# print(mi)
# mi /= np.max(mi)
#
# plt.figure(figsize=(15,3))
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     plt.scatter(x[:,i],y,edgecolors='black',s=20)
#     plt.xlabel("$x_{}$".format(i+1),fontsize=14)
#     if i==0:
#         plt.ylabel("$y$",fontsize=14)
#     plt.title("F_test={:.2f},MI={:.2f}".format(f_test[i],mi[i]),fontsize=16)
# plt.show()


# feature selection using Selected from model and lassoCV
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.datasets import load_boston
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LassoCV
# boston = load_boston()
# x,y = boston.data,boston.target
# clf = LassoCV()
# sfm = SelectFromModel(clf,threshold=0.25)
# sfm.fit(x,y)
# n_features = sfm.transform(x).shape[1]
# # print(sfm.transform(x).shape)
# # print(n_features)
# while n_features >2:
#     sfm.threshold += 0.1
#     x_transform = sfm.transform(x)
#     n_features = x_transform.shape[1]
# plt.title("feature selected from boston using selectFromModel with threshold %0.3f."%sfm.threshold)
# feature1 = x_transform[:,0]
# feature2 = x_transform[:,1]
# plt.plot(feature1,feature2,'r.')
# plt.xlabel('feature number 1')
# plt.ylabel('feature number 2')
# plt.ylim([np.min(feature2),np.max(feature2)])
# plt.show()

#l1 based feature selected
# from sklearn.svm import LinearSVC
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectFromModel
# iris = load_iris()
# x,y = iris.data,iris.target
# print(x.shape)
# lsvc = LinearSVC(C=0.01,penalty='l1',dual=False).fit(x,y)
# model = SelectFromModel(lsvc,prefit=True)
# x_new = model.transform(x)
# print(x_new.shape)
