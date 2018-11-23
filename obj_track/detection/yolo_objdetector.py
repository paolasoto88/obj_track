"""Run a YOLO_v2 style detection model on test images."""
import argparse
import colorsys
import imghdr
import os
import random
import cv2

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from ..yad2k.models.keras_yolo import yolo_eval, yolo_head
from .utils import read_classes, read_anchors, generate_colors, \
    preprocess_image, draw_boxes



def main(params):
    # todo-paola: delete the following line when executing from root directory
    os.chdir("..")

    # todo-paola: select yolo version.
    # Get proper files, model, anchors and labels.
    root_dir = os.getcwd()
    yolo_data_dir = os.path.join(root_dir, "models", "yolo", "data")
    model_path = None
    classes_path = None
    anchors_path = None
    files = os.listdir(yolo_data_dir)
    assert files, 'There are no files in yolo/data directory, ' \
                  'run convert_yad2k.py first'
    for file in files:
        if file.endswith('.h5'):
            model_path = os.path.join(yolo_data_dir, file)
        if file.endswith('classes.txt'):
            classes_path = os.path.join(yolo_data_dir, file)
        if file.endswith('anchors.txt'):
            anchors_path = os.path.join(yolo_data_dir, file)

    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    assert anchors_path.endswith('anchors.txt'), 'An *_anchors.txt file must ' \
                                                 'be provided'
    assert classes_path.endswith('classes.txt'), 'classes for dataset must be ' \
                                                 'provided in .txt file'
    # todo-paola: when args changed, change this accordingly
    test_path = os.path.expanduser(params.video)
    output_path = os.path.expanduser(params.save)
    os.makedirs(output_path, exist_ok=True)

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    class_names = read_classes(classes_path)

    anchors = read_anchors(anchors_path)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    colors = generate_colors(class_names)

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=0.3,
        iou_threshold=0.5)
    # boxes, scores, classes = yolo_eval(
    #     yolo_outputs,
    #     input_image_shape,
    #     score_threshold=params.score_threshold,
    #     iou_threshold=params.iou_threshold)

    for image_file in os.listdir(test_path):
        image, image_data = preprocess_image(os.path.join(test_path,
                                                          image_file),
                                             model_image_size, is_fixed_size)

        # Run the session, do prediction.
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for {}'.format(len(out_boxes), image_file))

        draw_boxes(image, out_scores, out_boxes, out_classes, class_names,
                   colors)
        image.save(os.path.join(output_path, image_file), quality=90)
    sess.close()