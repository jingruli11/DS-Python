from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
iris = load_iris()
X, y = iris.data, iris.target

classifier = KNeighborsClassifier()

# shuffle data before split
import numpy as np 
rng = np.random.RandomState(0)

permutation = rng.permutation(len(X))
X,y = X[permutation], y[permutation]
print(y)

# split data into training and testing
from sklearn.cross_validation import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X,y,train_size = 0.5, random_state = 1999)
print('Labels for training and testing data')
print(train_y)
print(test_y)

# evaluate classifier performance
classifier.fit(train_X,train_y)
pred_y = classifier.predict(test_X)
print('Fraction Correct')
print(np.sum(pred_y == test_y) / float(len(test_y)))

# visualize correct and failed predictions
import matplotlib.pyplot as plt 
correct_idx = np.where(pred_y == test_y)[0]
print(correct_idx)
incorrect_idx = np.where(pred_y != test_y)[0]
print(incorrect_idx)

# plot two dimensions
colors = ['darkblue', 'darkgreen', 'gray']
for n, color in enumerate(colors):
	idx = np.where(test_y == n)[0]
	plt.scatter(test_X[idx,0], test_X[idx,1], color = color, label = 'Class %s' % str(n))
plt.scatter(test_X[incorrect_idx,0], test_X[incorrect_idx,1], color = 'darkred')
# make xlim larger to accomodate legend
plt.xlim(3,9)
plt.legend(loc = 3)
plt.title('Iris Classification results')
plt.show()
