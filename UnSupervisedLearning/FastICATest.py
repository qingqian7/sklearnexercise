import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA,FastICA
from scipy import signal

np.random.seed(0)
n_samples = 2000
time = np.linspace(0,8,n_samples)
s1 = np.sin(2 * time)
s2 = np.sign(np.sin(3 * time))
s3 = signal.sawtooth(2 * np.pi * time)
s = np.c_[s1,s2,s3]
s += 0.2 * np.random.normal(size = s.shape)
s /= s.std(axis=0)

A = np.array([[1,1,1],[0.5,2,1],[1.5,1,2]])
X = np.dot(s,A.T)

ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)
A_= ica.mixing_
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

pca = PCA(n_components=3)
H = pca.fit_transform(X)

plt.figure()

models = [X,s,S_,H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']

for ii,(model,name) in enumerate(zip(models,names),1):
    plt.subplot(4,1,ii)
    plt.title(name)
    for sig,color in zip(model.T,colors):
        plt.plot(sig,color)
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()