import matplotlib.pyplot as plt 
import numpy as np 

# Unsupervised learning
from sklearn.datasets import load_iris
iris = load_iris()
X, y =iris.data, iris.target
print(X.shape)
print('mean: %s' % X.mean(axis = 0))
print('standard deviation: %s' % X.std(axis = 0))

# import and instantiate StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# only pass X not y, since this is an unsupervised learning
scaler.fit(X)

# rescaling data by using transform method, resulting 0 mean and unit stdev
X_scaled = scaler.transform(X)

print(X_scaled.shape)
print('mean: %s' % X_scaled.mean(axis = 0))
print('standard deviation: %s' % X_scaled.std(axis = 0))

# principle component analysis -- reduce dimensionality of data by creating a linear projection
# 2 dimensional data, Guassian blob rotated
rnd = np.random.RandomState(5)
X_ = rnd.normal(size = (300,2))
X_blob = np.dot(X_, rnd.normal(size = (2,2))) + rnd.normal(size = 2)
y = X_[:, 0] > 0
plt.scatter(X_blob[:,0],X_blob[:,1], c = y, linewidths = 0, s = 30)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()

# instantiate PCA model
from sklearn.decomposition import PCA
pca = PCA()

# fit PCA model with data, since unsupervised learning, no y output
pca.fit(X_blob)

# transform the data, projected on principal components
X_pca = pca.transform(X_blob)

plt.scatter(X_pca[:,0], X_pca[:,1], c = y, linewidths = 0, s = 30)
plt.xlabel('first principal component')
plt.ylabel('second principal component')
plt.title('PCA')
plt.show()

# manifold learning -- detect non-linear features
from sklearn.datasets import make_s_curve
X,y = make_s_curve(n_samples = 1000)
from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection = '3d')
ax.scatter3D(X[:,0], X[:,1], X[:,2], c = y)
ax.view_init(10,-60)
plt.show()

# using pca cannot discover underlying data orientation
X_pca = PCA(n_components =2).fit_transform(X)
plt.scatter(X_pca[:,0],X_pca[:,1], c = y)
plt.title('PCA')
plt.show()

# manifold learning algorithm, recover the underlying 2d manifold
from sklearn.manifold import Isomap
iso = Isomap(n_neighbors = 15, n_components = 2)
X_iso = iso.fit_transform(X)
plt.scatter(X_iso[:,0], X_iso[:,1], c = y)
plt.title('Manifold learning')
plt.show()




