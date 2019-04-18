import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
from network import Network, load_keys
import tensorflow as tf


plt.ion()


def plot(frame):
    data = float32_to_uint8(frame)
    im.set_data(data)
    fig.canvas.draw()
    fig.canvas.flush_events()


def uint8_to_float32(x):
    return tf.cast((x / 255) * 2 - 1, tf.float32)

def float32_to_uint8(x):
    return np.clip(255 * (x + 1) / 2, 0, 255).astype(np.uint8)


def get_image():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=1 / args.ratio, fy=1 / args.ratio)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


parser = argparse.ArgumentParser()
parser.add_argument("network_path", metavar="NETWORK_PATH", type=str, help="path to the net to be tested")
parser.add_argument('-r', '--ratio', type=int, help='downscale factor', default=1)
args = parser.parse_args()


cap = cv2.VideoCapture(0)


keys = load_keys(args.network_path + "/keys.txt")

first_image = get_image()
input_shape = first_image.shape
input_shape = [None] + list(input_shape)
net = Network(keys, input_shape)
inp = tf.placeholder(shape=input_shape, dtype=tf.uint8)
inp2 = uint8_to_float32(inp)
out = net(inp2)
loss = tf.reduce_mean((out - inp2) ** 2)

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(first_image)

saver = tf.train.Saver()
init_ops = []

with tf.Session() as sess:
    saver.restore(sess, args.network_path)
    while True:
        frame = np.expand_dims(get_image(), 0)
        nploss, rec, npinp = sess.run([loss, out, inp2], feed_dict={inp: frame})
        plot(rec[0])
        print(nploss, np.max(rec), np.max(npinp), end=" " * 20 + "\r")

cap.release()
