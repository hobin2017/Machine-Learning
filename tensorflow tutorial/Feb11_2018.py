"""
The code comes from the book 'Python机器学习及实践：从零开始通往Kaggle竞赛之路';
What is more, there is a module named skflow which is a combination of tensorflow and sci-kit learn;
"""

import tensorflow as tf
import numpy as np
import pandas as pd


# This will show how to use the basic element to form the ML model.
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)
linear = tf.add(product, tf.constant(2.0))

with tf.Session() as sess:
    result1 = sess.run(linear)
    print (result1)
print('---------------------------next example--------------------------------------------')

# This is the training in the linear regression.
train = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-train.csv')  # I do not have the data!
test = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-test.csv')  # I do not have the data!
X_train = np.float32(train[['Clump Thickness', 'Cell Size']].T)
y_train = np.float32(train['Type'].T)
X_test = np.float32(test[['Clump Thickness', 'Cell Size']].T)
y_test = np.float32(test['Type'].T)

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, X_train) + b
loss = tf.reduce_mean(tf.square(y - y_train))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for step in range(0, 1000):
    sess.run(train)
    if step % 200 == 0:
        print (step, sess.run(W), sess.run(b))


