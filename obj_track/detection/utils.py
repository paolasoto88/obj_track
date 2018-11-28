import numpy as np
import random
import colorsys
import imghdr
from PIL import Image, ImageDraw, ImageFont
from time import time as timer
import cv2


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

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def preprocess_image(image, model_image_size, fixed_size):
    if fixed_size:  # TODO: When resizing we can use minibatch input.
        resized_image = cv2.resize(image, tuple(reversed(model_image_size)),
                                   interpolation= cv2.INTER_CUBIC)
        image_data = np.array(resized_image, dtype='float32')
    else:
        # Due to skip connection + max pooling in YOLO_v2, inputs must have
        # width and height as multiples of 32.
        height, width, _ = image.shape
        new_image_size = (width - (width % 32), height - (height % 32))
        resized_image = cv2.resize(image, new_image_size,
                                   interpolation=cv2.INTER_CUBIC)
        image_data = np.array(resized_image, dtype='float32')
        print(image_data.shape)

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data


def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    font = 0
    fontSize = 1e-3 * image.shape[0]
    thickness = int((image.shape[1] + image.shape[0]) // 300)

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

        cv2.rectangle(image, (left, top), (right, bottom),
                      colors[c], thickness)
        cv2.putText(image, label, (left, top - 12), font, fontSize,
                    colors[c], thickness)