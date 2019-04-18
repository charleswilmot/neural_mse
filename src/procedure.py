from network import Network
import database
import argparse
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('input', metavar='INPUT', type=str, help='input data path')
parser.add_argument('-b', '--batch-size', type=int, default=32)
args = parser.parse_args()


def uint8_to_float32(x):
    return tf.cast((x / 255) * 2 - 1, tf.float32)

def float32_to_uint8(x):
    return np.clip(255 * (x + 1) / 2, 0, 255).astype(np.uint8)


dataset = database.get_dataset(args.input)
dataset = dataset.map(uint8_to_float32)
dataset = dataset.repeat()
#dataset = dataset.shuffle(5000)
dataset = dataset.batch(args.batch_size)
iterator = dataset.make_initializable_iterator()
inp = iterator.get_next()

keys = [
    "conv_outdim32_size8_stride4_act_lrelu",
    "conv_outdim64_size10_stride5_act_lrelu",
    "flatten",
    "fc_outdim500_act_lrelu",
    "fc_outdim768_act_lrelu",
    "reshape_6_8_16",
    "deconv_outdim64_size10_stride5_act_none",
    "deconv_outdim3_size8_stride4_act_none"
]

split_index = 4
encoder_keys, decoder_keys = keys[:split_index], keys[split_index:]
input_shape = inp.get_shape().as_list()

encoder = Network(encoder_keys, input_shape)
latent = encoder(inp)
input_shape = latent.get_shape().as_list()
decoder = Network(decoder_keys, input_shape)
out = decoder(latent)
loss = tf.reduce_mean((out - inp) ** 2)

latent_phd = tf.placeholder(shape=(None, 500), dtype=tf.float32)
inp_phd = tf.placeholder(shape=(None, 120, 160, 3), dtype=tf.float32)
out_decoder_only = decoder(latent_phd)
loss_decoder_only = tf.reduce_mean((out_decoder_only - inp_phd) ** 2)

optimizer = tf.train.AdamOptimizer(1e-3)
train_encoder = optimizer.minimize(loss, var_list=encoder.variables)
train_decoder = optimizer.minimize(loss_decoder_only, var_list=decoder.variables)
train_all = optimizer.minimize(loss)


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


with tf.Session() as sess:
    sess.run([iterator.initializer, tf.global_variables_initializer()])
    for i in range(0):
        nploss, nprecs, _ = sess.run([loss, out, train_all])
        print(nploss)
        im.set_data(float32_to_uint8(nprecs[0]))
        fig.canvas.draw()
        fig.canvas.flush_events()
    print("##############")
    for i in range(0):
        nploss, nprecs, _ = sess.run([loss, out, train_encoder])
        plot(nprecs[0])
        while nploss > 0.02:
            nploss_after, nprecs, _ = sess.run([loss, out, train_all])
            plot(nprecs[0])
            print(nploss, nploss_after)
            nploss = nploss_after
        else:
            print(nploss)
    print("##############")
    buffer_size = 2000 // args.batch_size
    previous_time = 0
    times_for_buffer_to_be_filled = []
    buffer_index = 0
    buf = np.zeros((buffer_size, args.batch_size * (120 * 160 * 3 + 500)))
    buf_inp = buf[:, :args.batch_size * 120 * 160 * 3].reshape((buffer_size, args.batch_size, 120, 160, 3))
    buf_latent = buf[:, args.batch_size * 120 * 160 * 3:].reshape((buffer_size, args.batch_size, 500))
    buf_loss = np.zeros(buffer_size)
    for i in range(5000000):
        nploss, npinp, nplatent, nprecs, _ = sess.run([loss, inp, latent, out, train_encoder])
        print(nploss)
        plot(0.95 * nprecs[0] + 0.05 * npinp[0])
        if nploss > 0.03:
            # store latent / inp pair in a queue
            buf_inp[buffer_index] = npinp
            buf_latent[buffer_index] = nplatent
            buffer_index += 1
            if buffer_index == buffer_size:
                # train decoder
                print("training decoder")
                current_time = i
                times_for_buffer_to_be_filled.append(current_time - previous_time)
                previous_time = current_time
                print(times_for_buffer_to_be_filled)
                np.random.shuffle(buf)
                for i, (npinp, nplatent) in enumerate(zip(buf_inp, buf_latent)):
                    nploss, nprecs, _ = sess.run([loss_decoder_only, out_decoder_only, train_decoder], feed_dict={latent_phd:nplatent, inp_phd:npinp})
                    buf_loss[i] = nploss
                    plot(0.95 * nprecs[0] + 0.05 * npinp[0])
                    print(nploss)
                arg = np.argsort(buf_loss)[::-1]
                buf = buf[arg]
                buf_loss = buf_loss[arg]
                buf_loss[:-1] = buf_loss[1:]
                print(buf_loss)
                buffer_index = 30
                print("training encoder")
