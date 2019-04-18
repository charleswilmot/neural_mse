import database
import cv2
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('output', metavar='OUTPUT', type=str, help='output data path')
parser.add_argument('-n', '--n-frames', type=int, help='number of frames to be generated', default=100)
parser.add_argument('-s', '--chunk-size', type=int, help='chunk size in bytes', default=500e6)
parser.add_argument('-r', '--ratio', type=int, help='downscale factor', default=1)
args = parser.parse_args()


cap = cv2.VideoCapture(0)


def get_image():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=1 / args.ratio, fy=1 / args.ratio)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


with database.DBWriter(args.output, chunk_size=args.chunk_size) as write:
    for i in range(args.n_frames):
        write(get_image())
        print(i, end='\r')
print("\n")

cap.release()
