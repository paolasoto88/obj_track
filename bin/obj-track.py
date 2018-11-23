import argparse
import json
from obj_track.detection.tf_objdetector_api import tfapi
from obj_track.detection.yolo_objdetector import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="object tracking module")
    parser.add_argument("-v", "--video", type=str, required=True,
                        help="path to input video file, 0 if webcam, url if "
                             "streaming")
    parser.add_argument("-d", "--detector", type=str, default="yolo",
                        help="object detector type, options: tfapi or yolo")
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
    elif args.detector == "yolo":
        #todo-paola: change args from Namespace to dict
        main(args)
    else:
        raise ValueError("Object detector type not valid. Valid options: "
                         "tfapi or yolo")