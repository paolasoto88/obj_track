import argparse

if __name__== 'main':
    parser = argparse.ArgumentParser(
        description="object tracking module")
    parser.add_argument("-cm", "--camera", dest="camera", action='store_true',
                        help="track from camera")
    parser.add_argument("-s", "--stream", dest="stream", action='store_true',
                        help="track from streaming")
    parser.add_argument("-f", "--file", dest="file", action='store_true',
                        help="track from video file")

    args = parser.parse_args()
