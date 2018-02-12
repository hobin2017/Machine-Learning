"""
An introduction to machine leaning with scikit-learn;
A dataset is a dictionary-like object;
In scikit-learn, an estimator for classification is a object with fit(X,y) function and predict(T)
It is given in P107-P110 (scikit-learn user guide, Release 0.19.1, Nov21 2017).
"""
from sklearn import datasets, svm
import pickle

"""step1: loading the dataset P108;"""
digits = datasets.load_digits()  # The project is the classification of number image ranging from 0 to 9;
# print(digits['data'])  # It is an array with shape (n_samples, n_features)
print(digits['data'].shape)
# print(digits['target'])  # It is the corresponding number for the image which is manually specified;
print(len(digits['target']))
print(digits['images'][0].shape)  # Each original sample is an image of shape (8, 8)


"""step2: using one estimator for the dataset P109"""
clf = svm.SVC(gamma=0.001, C=100.0)  # clf represents classifier
clf.fit(digits['data'][:-1], digits['target'][:-1])  # training the estimator in the training data;
print('The setting of the estimator is: \n%s' % clf)
print('The prediction is: %s' % clf.predict(digits['data'][-1:]))  # doing the prediction in the testing data;
print('''The testing data actually is the last image; And the array of this image is:\n %s''' % digits['data'][-1:])


"""
step3: one way to save the parameters of the estimator, hence no need to training again;
You can use another type of serialization   
You can use pickle.dump() to save it in file and then use pickle.load() to take it back;
"""
param_model = pickle.dumps(clf)  # It is binary data
clf2 = pickle.loads(param_model)  # loading the binary data
print('The setting of the estimator is: \n%s' % clf2)

