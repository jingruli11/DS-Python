import numpy as np

# Generating a random array
X = np.random.random((3, 5))  # a 3 x 5 array

print(X)
print(X[0, 0])

# get a row
print(X[1])

# get a column
print(X[:, 1])

# transpose matrix
print(X.T)

#turning a row vector into a column vector
y = np.linspace(0,12,5)
print(y)

#make into a column vector
print(y[:,np.newaxis])

#getting the shape and reshaping vector
print(X.shape)
print(X.reshape(5,3))

#indexing by an array of integers
indices = np.array([3,1,0])
print(indices)
print(X[:,indices])

from scipy import sparse

# create random array with zeroes
x = np.random.random((10,5))
print(x)

# set majority of elements equal to 0
x[x<0.7] = 0

#turn x into a CSR(Compressed-Sparsed-Row) matrix
# return nonzero element with coordinates first then value in a row
x_csr = sparse.csr_matrix(x)
print(x_csr)

# create an empty LIL (list in list) matrix and add some items
x_lil = sparse.	lil_matrix((5,5))
for i, j in np.random.randint(0,5,(15,2)):
	x_lil[i,j] = i + j

print(x_lil)
print(x_lil.toarray())

# convert lil matrix to CSR
print(x_lil.tocsr())

# Matplotlib
import matplotlib.pyplot as plt
# plotting a line
x = np.linspace(0,10,100)
plt.plot(x,np.sin(x))
plt.show()

# scatter plot points
x = np.random.normal(size = 500)
y = np.random.normal(size = 500)
plt.scatter(x,y)
plt.show()

# showing images
x = np.linspace(1,12,100)
y = x[:,np.newaxis]
im = y * np.sin(x) * np.cos(y)
print(im.shape)

# imshow-- origin at top-left by default
plt.imshow(im)
plt.show()

# contour plot -- origin at bottom-left by default
plt.contour(im)
plt.show()

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection = '3d')
xgrid, ygrid = np.meshgrid(x, y.ravel())
ax.plot_surface(xgrid,ygrid,im,cmap = plt.cm.jet, cstride = 2, rstride = 2, linewidth = 0)
plt.show()




