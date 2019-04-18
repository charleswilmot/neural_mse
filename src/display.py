import matplotlib.pyplot as plt
import database
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('input', metavar='INPUT', type=str, help='input data path')
args = parser.parse_args()


plt.ion()


fig = plt.figure()
ax = fig.add_subplot(111)
first = True

with database.DBReader(args.input) as frames:
    for frame in frames:
        if first:
            im = ax.imshow(frame)
            first = False
        else:
            im.set_data(frame)
            fig.canvas.draw()
            fig.canvas.flush_events()
