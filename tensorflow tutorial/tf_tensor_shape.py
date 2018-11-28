# -*- coding: utf-8 -*-
"""
Aim: to understand the tensor shape
"""

import tensorflow as tf

a = tf.Variable(list(range(24)))  # 2 * 2 * 3 * 2
a_reshaped = tf.reshape(a, [2, 2, 3, 2])

b = tf.Variable(list(range(12)))  # 1 * 2 * 3 * 2
b_reshaped = tf.reshape(b, [1, 2, 3, 2])

c = tf.Variable(list(range(5)))
c_reshaped = tf.reshape(c, [-1, 5])  # Is this array one-dimension or two-dimension? two dimension!
c_reshaped2 = tf.reshape(c, [-1])  # This array is one-dimension.

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(a)
    print('the original value is %s' % sess.run(a))
    print(a_reshaped)
    print('after reshaping, the value is %s' % sess.run(a_reshaped))
    print('---------------------------------------------')
    print(b)
    print('the original value is %s' % sess.run(b))
    print(b_reshaped)
    print('after reshaping, the value is %s' % sess.run(b_reshaped))
    print('---------------------------------------------')
    print(c)
    print('the original value is %s' % sess.run(c))
    print(c_reshaped)
    print('after reshaping, the value of c_reshaped is %s' % sess.run(c_reshaped))
    print('after reshaping, the value of c_reshaped2 is %s' % sess.run(c_reshaped2))

