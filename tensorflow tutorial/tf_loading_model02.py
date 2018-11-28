# -*- coding: utf-8 -*-
"""
Aim: loading models to do real time object detection
  Pre-works are
  1, setting the correct path for these two variables PATH_TO_CKPT and PATH_TO_LABELS;
  2, setting the correct value of NUM_CLASSES based on the total labels;
"""

import cv2
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
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

            total_times = 0
            while 1:
                ret, frame_rgb = cap.read()
                assert ret
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(frame_bgr, axis=0)

                # Actual detection.
                print('the model starts to detect with index %s' %total_times)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                print('the model ends to detect with index %s'%total_times)

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame_rgb,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                # Displaying the result by using opencv
                cv2.imshow('real time detection', frame_rgb)
                if cv2.waitKey(10) & 0xff == ord('q'):
                    break
                total_times = total_times + 1
    print('--------the detections ends--------')

