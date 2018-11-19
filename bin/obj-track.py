import argparse
import json
from obj_track.detection.tf_objdetector_api import tfapi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="object tracking module")
    parser.add_argument("-v", "--video", type=str, required=True,
                        help="path to input video file, 0 if webcam, url if "
                             "streaming")
    parser.add_argument("-d", "--detector", type=str, default="yolo",
                        required=True, help="object detector type, options: "
                                            "tfapi or yolo")
    parser.add_argument("-c", "--config", type=str,
                        help="path to the configuration of tfapi")

    args = parser.parse_args()
    if args.detector == "tfapi":
        if not args.config:
            raise ValueError("configuration of the TF API must be provided")
        with open(args.config, 'r') as f:
            tf_params = json.load(f)
        tf_params['video'] = args.video
        tfapi(tf_params)
    elif args.detector == "yolo":
        #todo-paola: add yolo implementation
        pass
    else:
        raise ValueError("Object detector type not valid. Valid options: "
                         "tfapi or yolo")