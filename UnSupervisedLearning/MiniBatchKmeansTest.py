import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

np.random.seed(0)
batch_size = 50
centers = [[1,1],[-1,-1],[1,-1]]
n_clusters = len(centers)
X,labels_true = make_blobs(n_samples=3000,centers=centers,cluster_std=0.7)

k_means = KMeans(init='k-means++',n_clusters=3,n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0

mbk = MiniBatchKMeans(init='k-means++',n_clusters=3,batch_size=batch_size,n_init=10,max_no_improvement=10,verbose=0)
t0 = time.time()
mbk.fit(X)
t_mini_batch = time.time() - t0

#plot the result
fig = plt.figure(figsize=(8,3))
fig.subplots_adjust(left=0.02,right=0.98,bottom =0.05,top = 0.9)
colors = ['#4EACC5','#FF9C34','#4E9A06']

k_means_cluster_centers = np.sort(k_means.cluster_centers_,axis=0)
mbk_cluster_centers = np.sort(mbk.cluster_centers_,axis=0)
k_means_labels = pairwise_distances_argmin(X,k_means_cluster_centers)
mbk_labels = pairwise_distances_argmin(X,mbk_cluster_centers)
print(mbk_labels)
order = pairwise_distances_argmin(k_means_cluster_centers,mbk_cluster_centers)
ax = fig.add_subplot(1,3,1)
for k, col in zip(range(n_clusters),colors):
    my_member = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_member,0],X[my_member,1],'w',markerfacecolor = col,marker='.')
    ax.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor = col,markeredgecolor='k',markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
    t_batch, k_means.inertia_))

ax = fig.add_subplot(1,3,2)
for k,col in zip(range(n_clusters),colors):
    my_members = mbk_labels == order[k]
    cluster_center = mbk_cluster_centers[order[k]]
    print(cluster_center[0])
    ax.plot(X[my_members,0],X[my_members,1],'w',markerfacecolor = col,marker='.')
    ax.plot(cluster_center[0],cluster_center[1],'o',markerfacecolor=col,markeredgecolor='k',markersize=6)
ax.set_title('MiniBatchKMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
         (t_mini_batch, mbk.inertia_))

#initialise the different array to all false
different = (mbk_labels == 4)
print(different)
ax = fig.add_subplot(1,3,3)
for k in range(n_clusters):
    different += ((k_means_labels == k) != (mbk_labels == order[k]))
identic = np.logical_not(different)
ax.plot(X[identic,0],X[identic,1],'w',markerfacecolor='#bbbbbb',marker='.')
ax.plot(X[different,0],X[different,1],'o',markerfacecolor='m',marker='.')
ax.set_title('different')
ax.set_xticks(())
ax.set_yticks(())
plt.show()