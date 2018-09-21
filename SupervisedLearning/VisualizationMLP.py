import io
from scipy.io.arff import loadarff
import matplotlib.pyplot  as plt
from sklearn.datasets import get_data_home
from sklearn.externals.joblib import Memory
from sklearn.neural_network import MLPClassifier
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

memory = Memory(get_data_home())
@memory.cache()
def fetch_mnist():
    content = urlopen('https://www.openml.org/data/download/52667/mnist_784.arff').read()
    data,meta = loadarff(io.StringIO(content.decode('utf8')))
    data = data.view([('pixels', '<f8', 784), ('class', '|S1')])
    return data['pixels'],data['class']
x,y = fetch_mnist()
x_train, x_test = x[:6000],x[6000:]
y_train, y_test = y[:6000],y[6000:]
mlp = MLPClassifier(hidden_layer_sizes=(50,),max_iter=10,alpha=1e-4,solver='sgd',verbose=10,tol=1e-4,random_state=1,learning_rate_init=.1)
mlp.fit(x_train,y_train)
print("Training set score :%f"%mlp.score(x_train,y_train))
print("Test set score:%f"%mlp.score(x_test,y_test))
fig,axes = plt.subplots(4,4)
vmin, vmax = mlp.coefs_[0].min(),mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

