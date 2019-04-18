import time
import datetime
from network import Network, save_keys
import database
import argparse
import tensorflow as tf
import os


parser = argparse.ArgumentParser()
parser.add_argument('input', metavar='INPUT', type=str, help='input data path')
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-n', '--n-batches', type=int, default=2000)
parser.add_argument('-o', '--output-path', type=str, default="../data_out/networks")
parser.add_argument('-r', '--restore-net', type=str, default=None)

args = parser.parse_args()


def uint8_to_float32(x):
    return tf.cast((x / 255) * 2 - 1, tf.float32)

def float32_to_uint8(x):
    return np.clip(255 * (x + 1) / 2, 0, 255).astype(np.uint8)


dataset = database.get_dataset(args.input)
dataset = dataset.map(uint8_to_float32)
dataset = dataset.shuffle(5000)
dataset = dataset.repeat()
dataset = dataset.batch(args.batch_size)
iterator = dataset.make_initializable_iterator()
inp = iterator.get_next()

keys = [
    "conv_outdim32_size8_stride4_act_tanh",
    "conv_outdim64_size10_stride5_act_tanh",
    "flatten",
    "fc_outdim500_act_tanh",
    "fc_outdim768_act_tanh",
    "reshape_6_8_16",
    "deconv_outdim64_size10_stride5_act_tanh",
    "deconv_outdim3_size8_stride4_act_none"
]

input_shape = inp.get_shape().as_list()

net = Network(keys, input_shape)
out = net(inp)
loss = tf.reduce_mean((out - inp) ** 2)

optimizer = tf.train.AdamOptimizer(1e-3)
train = optimizer.minimize(loss)

import matplotlib.pyplot as plt
import numpy as np

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.random.uniform(size=(120, 160, 3), low=0, high=255).astype(np.uint8))

def plot(frame):
    im.set_data(float32_to_uint8(frame))
    fig.canvas.draw()
    fig.canvas.flush_events()


saver = tf.train.Saver()
init_ops = [iterator.initializer, tf.global_variables_initializer()]
if args.restore_net is not None:
    init_ops.append(saver.load(args.restore_net))

with tf.Session() as sess:
    sess.run(init_ops)
    for i in range(args.n_batches):
        nploss, nprecs, _ = sess.run([loss, out, train])
        print(nploss)
        plot(nprecs[0])

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    path = args.output_path + "/{}/".format(st)
    os.mkdir(path)
    saver.save(sess, path)
    save_keys(path + "/keys.txt", keys)
