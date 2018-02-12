"""
SVM can be used in regression(SVR) or in classification(SVC)
it is given in P123 (scikit-learn user guide, Release 0.19.1, Nov21 2017).
"""
from sklearn import datasets, svm
import numpy as np


"""step 1: loading the data"""
iris = datasets.load_iris()  # it is dictionary-like object.
iris_X = iris.data
iris_y = iris.target

"""step 2: separating the data set"""
iris_X_train = iris_X[:-20]
iris_y_train = iris_y[:-20]
iris_X_test = iris_X[-20:]
iris_y_test = iris_y[-20:]

"""step 3: using a estimator"""
svc = svm.SVC(kernel='linear')
svc.fit(iris_X_train, iris_y_train)

"""step 4: estimating the model"""
error = np.mean((svc.predict(iris_X_test)-iris_y_test)**2)  # the mean square error. 0 means perfect!
print('the mean square error is: %s' % error)

