"""
Run a YOLO_v2 style detection model on video.
Modified version of
https://github.com/allanzelener/YAD2K/blob/master/test_yolo.py

"""
import os
import cv2

from keras import backend as K
from keras.models import load_model

from ..yad2k.models.keras_yolo import yolo_eval, yolo_head
from .utils import read_classes, read_anchors, generate_colors, \
    preprocess_image, draw_boxes, get_video_props

def yolo(params):
    """
    Object detection using YOLO implementation in Keras.

    Parameters
    ----------
    :param params: Namespace (must be changed to dict), used
    :return:
    """
    # todo-paola: delete the following line when executing from root directory
    os.chdir("..")

    # -----------------------------------------------------------------------#
    #  Get and load converted yolo model with anchors and classes and labels #
    # -----------------------------------------------------------------------#
    # Get proper files, model, anchors and labels.
    root_dir = os.getcwd()
    yolo_data_dir = os.path.join(root_dir, "models", "yolo", "data")
    model_path = None
    classes_path = None
    anchors_path = None
    files = os.listdir(yolo_data_dir)
    assert files, 'There are no files in yolo/data directory, ' \
                  'run convert_yad2k.py first'
    yolo_version = params.detector
    model_name = yolo_version + '.h5'
    anchors_name = yolo_version + '_anchors.txt'
    for file in files:
        if file == model_name:
            model_path = os.path.join(yolo_data_dir, file)
        if file.endswith('classes.txt'):
            classes_path = os.path.join(yolo_data_dir, file)
        if file == anchors_name:
            anchors_path = os.path.join(yolo_data_dir, file)

    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    assert anchors_path.endswith('anchors.txt'), 'An *_anchors.txt file must ' \
                                                 'be provided'
    assert classes_path.endswith('classes.txt'), 'classes for dataset must be ' \
                                                 'provided in .txt file'
    # todo-paola: when args changed, change this accordingly
    file = os.path.expanduser(params.video)
    output_path = os.path.expanduser(params.save)
    os.makedirs(output_path, exist_ok=True)
    # -----------------------------------------------------------------------#
    # -----------------------------------------------------------------------#

    # -----------------------------------------------------------------------#
    #           Configure TF session, load model and get properties          #
    # -----------------------------------------------------------------------#
    # Create TF session, generate classes and colors
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.
    class_names = read_classes(classes_path)
    anchors = read_anchors(anchors_path)
    colors = generate_colors(class_names)

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

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=0.3,
        iou_threshold=0.5)
    # -----------------------------------------------------------------------#
    # -----------------------------------------------------------------------#

    # -----------------------------------------------------------------------#
    #                           Configure Opencv                             #
    # -----------------------------------------------------------------------#
    # Get video file
    if file == '0':
        file = 0

    cap = cv2.VideoCapture(file)

    # if http address is not reachable assertion error will be raised
    assert cap.isOpened(), 'Cannot capture source'

    # todo-paola: add show option to parser
    show = True

    if show:
        cv2.namedWindow('demo', 0)
        _, frame = cap.read()
        height, width, _ = frame.shape
        cv2.resizeWindow('demo', 640, 480)

    # Get video properties
    fps, height, width = get_video_props(cap, file)

    if params.save:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_filename = output_path + '/output.avi'
        out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))


    while cap.isOpened():
        ret, image = cap.read()
        if image is None:
            print('\nEnd of Video')
            break
        image, image_data = preprocess_image(image, model_image_size,
                                             is_fixed_size)

        # Run the session, do prediction.
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_data,
                input_image_shape: [image.shape[0], image.shape[1]],
                K.learning_phase(): 0
            })
        #print('Found {} boxes for video'.format(len(out_boxes)))

        draw_boxes(image, out_scores, out_boxes, out_classes, class_names,
                   colors)
        if params.save:
            out.write(image)
        if show:
            cv2.imshow('demo', image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                cap.release()
                break
    sess.close()
    print('Job finished')
    cap.release()
    if show:
        cv2.destroyAllWindows()
    if params.save:
        out.release()