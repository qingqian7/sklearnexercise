from time import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

np.random.seed(40)
digits = load_digits()
data = scale(digits.data)
print(data.shape)
n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target
sample_size = 300
print("n_digits:%d,\t  n_samples %d, \t n_features %d" % (n_digits,n_samples,n_features))
print(70*'_')
print('init\t\ttime\t\tinertia\thomo\tcompl\tv-means\tARI\tAMI\tsilhouette ')
def bench_k_means(estimator,name,data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
bench_k_means(KMeans(init='k-means++',n_clusters=n_digits,n_init=10),name='k-means++',data = data)
bench_k_means(KMeans(init='random',n_clusters=n_digits,n_init=10),name="random",data=data)

#in this case the seeding of the centers is deterministic,hence we run the kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_,n_clusters=n_digits,n_init=1),name="pca-based",data= data)
print(70*'_')
#visualize the results of the pca reduced data
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++',n_clusters=n_digits,n_init=10)
kmeans.fit(reduced_data)

h = 0.2
x_min, x_max = reduced_data[:,0].min()-1,reduced_data[:,0].max()+1
y_min, y_max = reduced_data[:,1].min()-1,reduced_data[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
z = kmeans.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(z,interpolation='nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),cmap=plt.cm.Paired,aspect='auto',origin='lower')
plt.plot(reduced_data[:,0],reduced_data[:,1],'k.',markersize=2)
#plt.plot(data[:,0],data[:,1],data[:,2],'k.',markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1],marker="x",s=169,linewidths=3,color='w',zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()