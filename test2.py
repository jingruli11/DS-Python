from sklearn.datasets import load_iris
iris = load_iris()
print(iris.keys())

n_samples, n_features = iris.data.shape
print(n_samples)
print(n_features)
# the sepal length, sepal width, petal length and petal width of the first sample (first flower)
print(iris.feature_names)
print(iris.data[0])

print(iris.data.shape)
print(iris.target.shape)
print(iris.target)
print(iris.target_names)

import matplotlib.pyplot as plt
x_index = 3
y_index = 0

#formatter to label the colorbar with correct names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

#compare first and second features
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c = iris.target)
plt.colorbar(ticks = [0,1,2], format = formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.show()

# digits dataset
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.keys())
n_samples , n_features = digits.data.shape
print((n_samples,n_features))
print(digits.data[0])
print(digits.target)
print(digits.target_names)
print(digits.data.shape)
print(digits.images.shape)

import numpy as np
print(np.all(digits.images.reshape((1797,64)) == digits.data))

# set up the figure
fig = plt.figure(figsize = (6,6)) # figsize in inches
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

#plot the digits: each image is an 8*8 pixel
for i in range(64):
	ax = fig.add_subplot(8,8,i+1,xticks = [], yticks = [])
	ax.imshow(digits.images[i], cmap = plt.cm.binary, interpolation = 'nearest')
	# label the images witg target value
	ax.text(0,7,str(digits.target[i]))
plt.show()


# nonlinear dataset: the S-curve
from sklearn.datasets import make_s_curve
data, color = make_s_curve(n_samples = 1000)
print(data.shape)
print(color.shape)

from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection = '3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c = color)
ax.view_init(10,-60)
plt.show()


# faces dataset
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()

# set up the figure
fig = plt.figure(figsize = (6,6)) 	#figure size in inches
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)

# plot the faces
for i in range(64):
	ax = fig.add_subplot(8,8,i+1,xticks = [], yticks = [])
	ax.imshow(faces.images[i], cmap = plt.cm.binary, interpolation = 'nearest')

plt.show()

