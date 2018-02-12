"""
Linear Regression
If the desired output consists of one or more continuous variables, then the task is called regression P107.
it is given in P118(scikit-learn user guide, Release 0.19.1, Nov21 2017).
"""

from sklearn import linear_model, datasets
import numpy as np

"""step1: loading the data"""
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data  # It contains 442 samples
diabetes_y = diabetes.target

"""step2: splitting the data into training data and test data"""
diabetes_X_train = diabetes_X[:-20]
diabetes_y_train = diabetes_y[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_test = diabetes_y[-20:]
"""step3: using one estimator"""
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_test, diabetes_y_test)

"""step 4: estimating the model"""
error = np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)  # the mean square error. 0 means perfect!
print('the mean square error is: %s' % error)
relationship = regr.score(diabetes_X_test, diabetes_y_test)
# variance score, 1 means perfect prediction and 0 means that there is no linear relationship between X and y.
print('THe variance score is: %s' % relationship)

