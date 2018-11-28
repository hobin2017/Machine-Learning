# -*- coding: utf-8 -*-
"""
Aim: loading models to detection objects
  Pre-works are
  1, downloading the 'object_detection' folder from https://github.com/tensorflow/models and the folder sits in
the 'research' folder;
  2, downloading two files 'frozen_inference_graph.pb' and 'label_map.pbtxt';
  3, setting the correct path for these two variables PATH_TO_CKPT and PATH_TO_LABELS;
  4, setting the correct value of NUM_CLASSES based on the total labels;
  The code is modified based on the code of this book in p100: <21个项目玩转深度学习-基于TensorFlow的实践详解 (author何之源)>
"""
import os
import sys
import tarfile

import cv2
import numpy as np
import requests
import six.moves.urllib
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def pre_work(download_needed_status=0, file_extraction_needed_status=0):
    print('--------start to do the pre-work--------')
    # downloading large file
    MODEL_NAME01 = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_FILE = MODEL_NAME01 + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    if download_needed_status:
        print('start to download the model file')
        resp = requests.get(DOWNLOAD_BASE + MODEL_FILE, stream=True)  # reading contents in chunk
        if resp.status_code == 200:
            print('start to write them into file')
            with open(r'D:\%s' % MODEL_FILE, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=10240):
                    if chunk:
                        # print(type(chunk))
                        # print(len(chunk))
                        # print(chunk)
                        f.write(chunk)
        else:
            print('the status code of the response is not 200.')
        resp.close()  # you need to close it manually since you set the stream parameter to be True.

        # the original way to download files
        # opener = six.moves.urllib.request.URLopener()
        # opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        print('end to download the model file')

    # file extraction
    if file_extraction_needed_status:
        print('start to extra the model file')
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                current_working_directory = os.getcwd()
                tar_file.extract(file, current_working_directory)
        print('end to extra the model file')
    print('--------end to do the pre-work--------')


if __name__ == '__main__':
    # downloading the zip/tar/rar file and extracting them
    prework_needed_status = 0
    download_needed_status = 0
    file_extraction_needed_status = 0
    if prework_needed_status:
        pre_work(download_needed_status=download_needed_status,
                 file_extraction_needed_status=file_extraction_needed_status)
        sys.exit(0)

    print('--------start to load the model file--------')
    PATH_TO_CKPT = r'.\ssd_mobilenet_v1_coco_11_06_2017\frozen_inference_graph.pb'
    PATH_TO_LABELS = r'.\ssd_mobilenet_v1_coco_11_06_2017\mscoco_label_map.pbtxt'
    NUM_CLASSES = 90

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print('--------end to load the model file--------')

    print('--------the detections starts--------')
    # adding path to the images to the TEST_IMAGE_PATHS to detect your images
    TEST_IMAGE_PATHS = [r'./images/Lion.png']
    IMAGE_SIZE = (12, 8) # Size, in inches, of the output images.

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                image_np = cv2.imread(image_path)  # ndarray type

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                # Actual detection.
                print('the model starts to detect')
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                print('the model ends to detect')

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                # Displaying the result by using opencv
                cv2.imshow('final result', image_np)
                cv2.waitKey(0)
    print('--------the detections ends--------')

