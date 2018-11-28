# -*- coding: utf-8 -*-
"""

"""

import tensorflow as tf

a = tf.Variable(list(range(6)))
a1 = tf.reshape(a, [2, 3])


b = tf.Variable([3, 3, 3])
# print(b.shape) # （3, ） indicating 3 columns


c = tf.Variable(list(range(27)))
c1 = tf.reshape(c, [3, 3, 3])
c2 = tf.reshape(c, [9, 3])
c3 = tf.reshape(c, [3, 9])  # the addition of c3 and b will raise error.


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print('Array a1 is %s.' % sess.run(a1))
    print('Array b is %s.' % sess.run(b))
    print('The addition of a1 and b is %s.' % sess.run(a1 + b))
    print('------------------------------------------------------------')
    print('Array c1 is %s' % sess.run(c1))
    print('Array b is %s.' % sess.run(b))
    print('Array c2 is %s' % sess.run(c2))
    print('The addition of c1 and b is %s.' % sess.run(c1 + b))
    print('The addition of c2 and b is %s.' % sess.run(c2 + b))
    print('------------------------------------------------------------')

