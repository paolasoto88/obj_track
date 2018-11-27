import argparse
import json
from obj_track.detection.tf_objdetector_api import tfapi
from obj_track.detection.yolo_v2_objdetector import yolo_v2

if __name__ == "__main__":
    """
    Main function to achieve object tracking in one video feed. Idea taken from 
    https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
    Run this program like this:
    - python obj_detector.py [OPTIONS]
    
    OPTIONS
    -------
    -v, --video : specify the path to the video source, 0 for webcam, url for 
    streaming or complete file for video store locally. 
    -d, --detector : choose which of the detectors is used to create bounding 
    boxes, tfapi for tensorflow, yolov2 for YOLO. 
    -s, --save : specify the path where the predictions are stored.
    -c, --config : path to the config file used by the tensorflow api. 
    """
    parser = argparse.ArgumentParser(
        description="object tracking module")
    parser.add_argument("-v", "--video", type=str, required=True,
                        help="path to input video file, 0 if webcam, url if "
                             "streaming")
    parser.add_argument("-d", "--detector", type=str, default="yolov2",
                        help="object detector type, options: tfapi or yolov2")
    parser.add_argument("-s", "--save", type=str, help="save option, give a "
                                                       "path to store the "
                                                       "results")
    parser.add_argument("-c", "--config", type=str,
                        help="path to the configuration of tfapi")


    args = parser.parse_args()
    if args.detector == "tfapi":
        # todo-paola: implement save option in tf_api
        if not args.config:
            raise ValueError("configuration of the TF API must be provided")
        with open(args.config, 'r') as f:
            tf_params = json.load(f)
        tf_params['video'] = args.video
        tfapi(tf_params)
    elif args.detector.startswith("yolo"):
        #todo-paola: change args from Namespace to
        #todo-paola: add support for yolov3
        #todo-paola: include the convert script -d must include the version
        # of yolo
        yolo_v2(args)
    else:
        raise ValueError("Object detector type not valid. Valid options: "
                         "tfapi or yolov2")