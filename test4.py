import matplotlib.pyplot as plt 
import numpy as np 
# load dataset
from sklearn.datasets import make_blobs
X, y = make_blobs(centers = 2, random_state = 0)
print(X.shape)
print(y.shape)
print(X[:5, :])
print(y[:5])

# plot the 2 dimensional data
plt.scatter(X[:,0],X[:,1],c = y, s = 40)
plt.xlabel('First feature')
plt.ylabel('Second feature')
plt.show()

# split training and testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#Scikit learn estimator API
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
X_train.shape
y_train.shape

# fit function with training data and corresponding labels
classifier.fit(X_train,y_train)
# applyc model to test dataset
prediction = classifier.predict(X_test)
#compare prediction result against test dataset labels
print(prediction)
print(y_test)

# evaluate accuracy
print(np.mean(prediction == y_test))

# score function to evaluate performance
print('Performance of training dataset')
print(classifier.score(X_train,y_train))
print('Performance of testing dataset')
print(classifier.score(X_test,y_test))

# print estimate parameters in logistic regression
print(classifier.coef_)
print(classifier.intercept_)


# K nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
# fit and plot knn model
knn.fit(X_train,y_train)
plt.scatter(X[:,0],X[:,1],c = y, s = 40)
plt.xlabel('First feature')
plt.ylabel('Second feature')
plt.show()
print(knn.score(X_test,y_test))


# Using knn to evaluate iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
print('Training dataset performance')
print(knn.score(X_train,y_train))
print('Testing dataset performance')
print(knn.score(X_test,y_test))

print('Prediction result')
print(knn.predict(X_test))






