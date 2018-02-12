"""
Problem: If you get a lots picture about flower, but you cannot label them (e.g. specify the flower name).
Solution: you can use clustering to label them automatically.
The simplest clustering algorithm is K-means;
Every well-separated group is called cluster;
it is given in P130 (scikit-learn user guide, Release 0.19.1, Nov21 2017).
"""
from sklearn import cluster, datasets
"""step1"""
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

"""step2"""
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(iris_X)

"""step3"""
print('The automatically labeled result is %s' % k_means.labels_[::10])
print('The manually labeled result is %s' % iris_y[::10])

