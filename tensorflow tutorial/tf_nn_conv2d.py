# -*- coding: utf-8 -*-
"""
I always think the padding applies after the convolution but the padding applies before the convolution.
test1() is about one channel convolution while test() is about three channel convolution.
np.array_equal():
    Returns True if two arrays have the same shape and elements, False otherwise.
In tf.nn.conv2d(), a kernel/filter tensor with shape [filter_height, filter_width, in_channels, out_channels]
In tf.nn.conv2d(), an input tensor with shape [batch, in_height, in_width, in_channels]
"""
import numpy as np
import cv2
import tensorflow as tf

def test01(origin_img, img_h, img_w):
    """
    one channel convolution
    :param origin_img:
    :param img_h:
    :param img_w:
    :return:
    """
    print('-----------------1 Using only one channel of the original image------------------------------------')
    lion_arr_1channel = origin_img[:, :, 0]

    print('-----------------Doing the convolution by for loop------------------------------------')
    padded_array = np.pad(lion_arr_1channel, (1, 1), 'constant')
    print('The shape of padded array is %s.' % str(padded_array.shape))  # (305, 499)
    kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    print('The shape of kernel is %s.' % str(kernel.shape))  # (3, 3)
    output_array01 = np.zeros((img_h, img_w))
    # starting the convolution
    for current_row in range(padded_array.shape[0] - 2):
        for current_column in range(padded_array.shape[1] - 2):
            temp_array = padded_array[current_row:current_row + 3, current_column: current_column + 3]
            # print(temp_array.shape)  # It should be (3, 3)
            # Be careful about the value which might not be inside [0, 255].
            output_array01[current_row, current_column] = np.sum(temp_array * kernel)

    print('-----------------Doing the convolution with padding by using TensorFlow-----------------------')
    # preparing kernel/filter tensor with shape [filter_height, filter_width, in_channels, out_channels]
    kernel_for_tf01 = np.array(
        [  # This bracket is for array
            [  # This is the first row;
                [[0]], [[0]], [[0]]
            ],
            [  # This is the second row;
                [[0]], [[1]], [[0]]
            ],
            [  # This is the third row;
                [[0]], [[0]], [[0]]
            ]
        ]
    )

    # preparing the input tensor with shape [batch, in_height, in_width, in_channels]
    lion_for_tf = lion_arr_1channel.reshape(1, img_h, img_w, 1)

    # starting the calculation
    graph = tf.Graph()
    with graph.as_default():
        tf_input_image = tf.Variable(lion_for_tf.astype(np.float32))
        print(type(tf_input_image))
        print('The shape of input tensor is %s.' % str(tf_input_image.shape))  # (1, 303, 497, 1)
        tf_kernel01 = tf.Variable(kernel_for_tf01.astype(np.float32))
        print(type(tf_kernel01))
        print('The shape of kernel tensor is %s' % str(tf_kernel01.shape))  # (3, 3, 1, 1)
        tf_convolution_output = tf.nn.conv2d(tf_input_image, tf_kernel01, strides=[1, 1, 1, 1], padding='SAME')
        print(type(tf_convolution_output))

    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        transformed_image = tf_convolution_output.eval()  # <class 'numpy.ndarray'>
        print('The shape of array after tf.nn.conv2d() is %s.' % str(transformed_image.shape))
        output_array2 = transformed_image[0, :, :, 0]

    print('--------------Verifying these two convolution output.--------------------------------')
    print(np.array_equal(output_array01, output_array2))
    np.testing.assert_array_almost_equal(output_array01, output_array2, decimal=4)


def test02(origin_img, img_h, img_w):
    """
    three channel convolution.
    :param origin_img:
    :param img_h:
    :param img_w:
    :return:
    """
    print('-----------------2 Using three channels of the original image------------------------------------')
    print('-----------------Doing the convolution by for loop------------------------------------')
    padded_array = np.pad(origin_img, (1, 1), 'constant')  # The third dimension is also padded.
    print('The shape of padded array is %s.' % str(padded_array.shape))  # (305, 499, 5)
    kernel = np.array([
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    ])
    print('The shape of kernel is %s.' % str(kernel.shape))  # (3, 3, 3)
    output_array01 = np.zeros((img_h, img_w))
    # starting the convolution
    for current_row in range(padded_array.shape[0] - 2):
        for current_column in range(padded_array.shape[1] - 2):
            temp_array = padded_array[current_row:current_row + 3, current_column: current_column + 3, 1:4]
            output_array01[current_row, current_column] = np.sum(temp_array * kernel)

    print('-----------------Doing the convolution with padding by using TensorFlow-----------------------')
    # preparing kernel/filter tensor with shape [filter_height, filter_width, in_channels, out_channels]
    kernel_for_tf01 = np.array(
        [  # This bracket is for array
            [  # This is the first row;
                [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]
            ],
            [  # This is the second row;
                [[0], [0], [0]], [[1], [1], [1]], [[0], [0], [0]]
            ],
            [  # This is the third row;
                [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]
            ]
        ]
    )

    # preparing the input tensor with shape [batch, in_height, in_width, in_channels]
    lion_for_tf = origin_img.reshape(1, img_h, img_w, 3)
    # starting the calculation
    graph = tf.Graph()
    with graph.as_default():
        tf_input_image = tf.Variable(lion_for_tf.astype(np.float32))
        print(type(tf_input_image))
        print('The shape of input tensor is %s.' %str(tf_input_image.shape))  # (1, 303, 497, 3)
        tf_kernel01 = tf.Variable(kernel_for_tf01.astype(np.float32))
        print(type(tf_kernel01))
        print('The shape of kernel tensor is %s' % str(tf_kernel01.shape))  # (3, 3, 3, 1)
        tf_convolution_output = tf.nn.conv2d(tf_input_image, tf_kernel01, strides=[1, 1, 1, 1], padding='SAME')
        print(type(tf_convolution_output))

    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        transformed_image = tf_convolution_output.eval()  # <class 'numpy.ndarray'>
        print('The shape of array after tf.nn.conv2d() is %s.' % str(transformed_image.shape))
        output_array2 = transformed_image[0, :, :, 0]

    print('--------------Verifying these two convolution output.--------------------------------')
    print(np.array_equal(output_array01, output_array2))
    np.testing.assert_array_almost_equal(output_array01, output_array2, decimal=4)

if __name__ == '__main__':

    lion_arr_original = cv2.imread('./images/Lion.png')
    # Although the image is in grayscale, this image actually still has three channels (red, green, and blue).
    print('The shape of original image is %s.' % str(lion_arr_original.shape))  # (303, 497, 3)

    # Maybe all the color channels are actually the same?
    print('Comparing channel 0 with channel 1 and the result is %s.' 
          % np.array_equal(lion_arr_original[:, :, 0], lion_arr_original[:, :, 1]))
    print('Comparing channel 1 with channel 2 and the result is %s.' 
          % np.array_equal(lion_arr_original[:, :, 1], lion_arr_original[:, :, 2]))

    test01(origin_img=lion_arr_original, img_h=lion_arr_original.shape[0], img_w=lion_arr_original.shape[1])
    test02(origin_img=lion_arr_original, img_h=lion_arr_original.shape[0], img_w=lion_arr_original.shape[1])


