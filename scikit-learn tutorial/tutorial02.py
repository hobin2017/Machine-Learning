"""
KNN (K nearest neighbor) classifier
it is given in P115-P117 (scikit-learn user guide, Release 0.19.1, Nov21 2017).
"""
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

"""step 1: loading the data"""
iris = datasets.load_iris()  # it is dictionary-like object.
iris_X = iris.data  # it is ndarray type
# print(type(iris_X))
iris_y = iris.target  # it is ndarray type
# print(type(iris_y))
# print('the possible output value is %s' % np.unique(iris_y))

"""step 2: splitting randomly the data into training data and testing data"""
# rearrange the dataset
np.random.seed(0)
indices = np.random.permutation(150)  # the length of all data is 150
# print(indices)  # it is a list of numbers range from 0 to 149
iris_X_train = iris_X[indices[0:130]]
iris_y_train = iris_y[indices[0:130]]
iris_X_test = iris_X[indices[130:150]]
iris_y_test = iris_y[indices[130:150]]


"""step 3: using one estimator for the dataset"""
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
iris_y_predicted = knn.predict(iris_X_test)
accuracy = np.sum(iris_y_test == iris_y_predicted)/20
print('the accuracy of this prediction is: %s' % accuracy)
