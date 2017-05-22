import matplotlib.pyplot as plt 
import numpy as np 

# dataset of sin curve with noise
x = np.linspace(-3,3,100)
print(x)
rng = np.random.RandomState(42)
y = np.sin(4*x) + x + rng.uniform(size = len(x))
plt.plot(x,y,'o')
plt.title('Sin curve with noise')
plt.show()

# Linear regression
# first make x a 2d array
print(x.shape)
X = x[:,np.newaxis]
print(X.shape)

# split dataset into training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42)

# build regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predict using training dataset and plot
y_pred_train = regressor.predict(X_train)
plt.plot(X_train,y_train,'o',label = 'data')
plt.plot(X_train,y_pred_train,'o', label = 'prediction')
plt.legend(loc = 'best')
plt.title('Training prediction')
plt.show()

# predict using testing dataset and plot
y_pred_test = regressor.predict(X_test)
plt.plot(X_test,y_test,'o',label = 'data')
plt.plot(X_test,y_pred_test,'o', label = 'prediction')
plt.legend(loc = 'best')
plt.title('Testing prediction')
plt.show()


print(regressor.score(X_test, y_test))

# KNeighborsRegression
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 1)
knn.fit(X_train, y_train)

# predict using training dataset and plot
y_pred_train = knn.predict(X_train)
plt.plot(X_train,y_train,'o', label = 'data')
plt.plot(X_train, y_pred_train, 'o', label = 'prediction')
plt.legend(loc = 'best')
plt.title('Training prediction KNN')
plt.show()

# predict using testing dataset and plot
y_pred_test = knn.predict(X_test)
plt.plot(X_test,y_test,'o', label = 'data')
plt.plot(X_test, y_pred_test, 'o', label = 'prediction')
plt.legend(loc = 'best')
plt.title('Testing prediction KNN')
plt.show()

print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))



# analyze boston housing data using regression
from sklearn.datasets import load_boston
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target)
regressor.fit(X_train, y_train)


print(regressor.score(X_train, y_train))
print(regressor.score(X_test, y_test))

# analyze using KNN regression
knn = KNeighborsRegressor(n_neighbors = 3)
knn.fit(X_train, y_train)


print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))


