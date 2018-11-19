OBJECTS_INTEREST = ['person', 'bicycle', 'car']

DATASETS={'coco': 'mscoco_label_map.pbtxt', 'kitti': 'kitti_label_map.pbtxt',
          'open_images': 'oid_object_detection_challenge_500_label_map.pbtxt',
          'inatur': 'fgvc_2854_classes_label_map.pbtxt',
          'ava': 'ava_label_map_v2.1.pbtxt'}

# What model to download.
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'