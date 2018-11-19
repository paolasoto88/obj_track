import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from time import time as timer
import json

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import  Image

sys.path.append("..")
from models.research.object_detection.utils import ops as utils_ops
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* '
                      'or later!')

from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as \
    vis_util
from obj_track.detection.const import DATASETS, DOWNLOAD_BASE



def get_video_props(capture, file):
    fps = 30
    h = 600
    w = 800
    if file == 0:
        num_frames = 120
        start = timer()
        for i in range(num_frames):
            _, frame = capture.read()
            h, w, _ = frame.shape
        end = timer() - start
        fps = round(num_frames / end)
        if fps < 1:
            fps = 1
    else:
        fps = round(capture.get(cv2.CAP_PROP_FPS))
        _, frame = capture.read()
        h, w, _ = frame.shape
    return fps, h, w


def tfapi(params):
    print('Running TensorFlow detector on video')
    # -----------------------------------------------------------------------#
    #          tensorflow configuration, load a pre-trained model            #
    # -----------------------------------------------------------------------#

    print('Configuring TensorFlow model')
    # List of the strings that is used to add correct label for each box.
    BASE_DIR = params['base_dir'] + '/data'
    DATASET_LABEL_MAP = DATASETS[params['dataset']]
    PATH_TO_LABELS = os.path.join(BASE_DIR, DATASET_LABEL_MAP)
    MODEL_NAME = params['model_name']
    MODEL_FILE = MODEL_NAME + '.tar.gz'

    # Path to frozen detection graph. This is the actual model that is used
    # for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

    # Download model
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for member in tar_file.getmembers():
        file_name = os.path.basename(member.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(member, os.getcwd())

    # Load a (frozen) TensorFlow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=True)
    # -----------------------------------------------------------------------#
    # -----------------------------------------------------------------------#

    # -----------------------------------------------------------------------#
    #                  Video Settings, opencv-python                         #
    # -----------------------------------------------------------------------#
    file = params['video']
    save = params['save']
    if file == "0":
        file = 0


    cap = cv2.VideoCapture(file)

    print('Press [q] to quit demo')

    #if url is not reachable assertion error will be raised
    assert cap.isOpened(), \
        'Cannot capture source'

    # Get video properties
    fps, height, width = get_video_props(cap, file)

    if save:
        # Create the output dir if doesn't exist
        os.makedirs(params['out'], exist_ok=True)
        video_filename = params['out'] + params['filename']
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    show = params['show']
    if show:
        cv2.namedWindow('demo', 0)
        _, frame = cap.read()
        height, width, _ = frame.shape
        cv2.resizeWindow('demo', 640, 480)
    # -----------------------------------------------------------------------#
    # -----------------------------------------------------------------------#



    # -----------------------------------------------------------------------#
    #       tensorflow configuration, run the prediction session             #
    # -----------------------------------------------------------------------#
    print('Prediction running')
    elapsed = int()
    # Running the tensorFlow session
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while cap.isOpened():
                ret, image_np = cap.read()
                if image_np is None:
                    print('\nEnd of Video')
                    break
                elapsed += 1
                if elapsed % params['num_frames'] ==0:
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name(
                        'image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name(
                        'detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name(
                        'num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    image_np = vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8, min_score_thresh=params['threshold'])
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # font_size = 1e-3 * height
                    # font_color = (255,255,255)
                    # image_np = cv2.putText(image_np,json.dumps(json_out),
                    #                        (10, 20), font, font_size, font_color, 2)
                    if save:
                        out.write(image_np)
                if show:
                    cv2.imshow('demo', image_np)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        cap.release()
                        break

    # Release everything if job is finished
    print('Job finished')
    cap.release()
    if save:
        out.release()
    if show:
        cv2.destroyAllWindows()