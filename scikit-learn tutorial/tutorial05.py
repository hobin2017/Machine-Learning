"""
the cross_val_score() splits the dataset repeatedly and automatically;
it is given in P127 (scikit-learn user guide, Release 0.19.1, Nov21 2017).
"""
from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score

"""step 1: loading the data"""
digits = datasets.load_digits()
digits_X = digits.data
digits_y = digits.target

"""step 2: separating the data set"""
k_fold = KFold(n_splits=4)  # splits it into K folds, trains on K-1 and then tests on the left-out;P127

"""step 3: using a estimator"""
svc = svm.SVC(C=1, kernel='linear')

"""step 4: estimating the model"""
scores = cross_val_score(svc, digits_X, digits_y, cv=k_fold)
print(scores)


