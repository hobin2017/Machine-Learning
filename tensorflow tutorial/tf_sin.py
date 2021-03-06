# coding=utf-8
'''
用tensorflow来拟合一个正弦函数，在predicting part中，扩大范围就发现它的预测和sin曲线不符了。
这是因为训练数据是0到2*pi
The code comes from the book: 《TensorFlow入门与实践》 and the author is 罗冬日.
'''
import pdb
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import types
import pylab


def draw_correct_line(x_start=0, x_end=2 * np.pi, step=0.01):
    '''
    绘制标准的sin的曲线
    '''
    x = np.arange(x_start, x_end, step)
    x = x.reshape((len(x), 1))
    y = np.sin(x)

    pylab.plot(x, y, label='标准sin曲线')
    plt.axhline(linewidth=1, color='r')


def get_train_data():
    '''
    返回一个训练样本(train_x, train_y),
    其中train_x是随机的自变量， train_y是train_x的sin函数值
    '''
    train_x = np.random.uniform(0.0, 2 * np.pi, (1))
    train_y = np.sin(train_x)
    return train_x, train_y


def inference(input_data):
    '''
    定义前向计算的网络结构
    Args:
        输入的x的值，单个值
    '''
    with tf.variable_scope('hidden1'):
        # 第一个隐藏层，采用16个隐藏节点
        weights = tf.get_variable("weight", [1, 16], tf.float32,
                                  initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("biase", [1, 16], tf.float32,
                                 initializer=tf.random_normal_initializer(0.0, 1))
        hidden1 = tf.sigmoid(tf.multiply(input_data, weights) + biases)

    with tf.variable_scope('hidden2'):
        # 第二个隐藏层，采用16个隐藏节点
        weights = tf.get_variable("weight", [16, 16], tf.float32,
                                  initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("biase", [16], tf.float32,
                                 initializer=tf.random_normal_initializer(0.0, 1))
        mul = tf.matmul(hidden1, weights)
        hidden2 = tf.sigmoid(mul + biases)

    with tf.variable_scope('hidden3'):
        # 第三个隐藏层，采用16个隐藏节点
        weights = tf.get_variable("weight", [16, 16], tf.float32,
                                  initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("biase", [16], tf.float32,
                                 initializer=tf.random_normal_initializer(0.0, 1))
        hidden3 = tf.sigmoid(tf.matmul(hidden2, weights) + biases)

    with tf.variable_scope('output_layer'):
        # 输出层
        weights = tf.get_variable("weight", [16, 1], tf.float32,
                                  initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("biase", [1], tf.float32,
                                 initializer=tf.random_normal_initializer(0.0, 1))
        output = tf.matmul(hidden3, weights) + biases

    return output


def train():
    # 学习率
    learning_rate = 0.01

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    net_out = inference(x)

    # 定义损失函数的op
    loss = tf.square(net_out - y)

    # 采用随机梯度下降的优化函数
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = opt.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("training starts")
        for i in range(150000):
            train_x, train_y = get_train_data()
            sess.run(train_op, feed_dict={x: train_x, y: train_y})

            # 每1万步画图看看效果
            # if i % 10000 == 0:
            #     times = int(i / 10000)
            #     test_x_ndarray = np.arange(0, 2 * np.pi, 0.01)  # starting from 0, incrementing by 0.01
            #     test_y_ndarray = np.zeros([len(test_x_ndarray)])
            #     ind = 0
            #     for test_x in test_x_ndarray:
            #         test_y = sess.run(net_out, feed_dict={x: test_x, y: 1})
            #         np.put(test_y_ndarray, ind, test_y)
            #         ind += 1
            #     # 先绘制标准的sin函数的曲线，
            #     # 再用虚线绘制我们计算出来模拟sin函数的曲线
            #     draw_correct_line()
            #     pylab.plot(test_x_ndarray, test_y_ndarray, '--', label=str(times) + 'times')
            #     pylab.show()
        print('training ends')

        # predicting
        predict_x_ndarray = np.linspace(-4*np.pi, 10 * np.pi, 1000)
        predict_y_ndarray = np.zeros([len(predict_x_ndarray)])
        for index, predict_x in enumerate(predict_x_ndarray):
            predict_y = sess.run(net_out, feed_dict={x: predict_x, y: np.sin(predict_x)})
            np.put(predict_y_ndarray, index, predict_y)
        draw_correct_line(-4*np.pi, 10 * np.pi, 0.01)
        pylab.plot(predict_x_ndarray, predict_y_ndarray)
        pylab.show()
        pdb.set_trace()
        print('session ends')


if __name__ == "__main__":
    train()
    pdb.set_trace()
