import numpy as np
import tensorflow as tf
import os
import re


def read_protobuf(shape):
    def read(example_proto):
        features = {"frame": tf.FixedLenFeature([], tf.string)}
        parsed_features = tf.parse_single_example(example_proto, features)
        data = tf.decode_raw(parsed_features["frame"], tf.uint8)
        return tf.reshape(data, shape)
    return read


def get_dataset(path):
    shape = np.load(path + "/shape.npy")
    regex = r'webcam_data_chunk([0-9]+).tfr'
    filenames = [x for x in os.listdir(path) if re.match(regex, x) is not None]
    filenames.sort(key=lambda n: int(re.match(regex, n).group(1)))
    filepaths = [path + '/' + x for x in filenames]
    dataset = tf.data.TFRecordDataset(filepaths)
    return dataset.map(read_protobuf(shape))


class DBWriter:
    def __init__(self, path, chunk_size=1e10):
        self.path = path
        self.chunk_size = chunk_size
        self._pattern = "webcam_data_chunk{}.tfr"
        self._chunk_number = 0
        self._current_size = 0
        self._shape_stored = False
        os.mkdir(path)
        self._writer = self._get_writer()

    def _get_writer(self):
        path = self.path + '/' + self._pattern.format(self._chunk_number)
        return tf.python_io.TFRecordWriter(path)

    def _serialize(self, frame):
        value = frame.astype(np.uint8).tobytes()
        feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        feature = {"frame": feature}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def _write(self, string):
        if self._current_size > self.chunk_size:
            self._writer.close()
            self._chunk_number += 1
            self._current_size = 0
            self._writer = self._get_writer()
        self._writer.write(string)
        self._current_size += len(string)

    def _store_shape(self, shape):
        if not self._shape_stored:
            np.save(self.path + "/shape.npy", shape)
            self._shape_stored = True

    def __call__(self, frame):
        self._store_shape(frame.shape)
        self._write(self._serialize(frame))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._writer.close()


class DBReader:
    def __init__(self, path):
        self.path = path
        self.frame_shape = np.load(path + "/shape.npy")
        self.dataset = get_dataset(path)
        self.iterator = self.dataset.make_initializable_iterator()
        self.frame = self.iterator.get_next()

    def __enter__(self):
        self._sess = tf.Session()
        return self

    def __exit__(self, type, value, traceback):
        self._sess.close()

    def __iter__(self):
        self._sess.run(self.iterator.initializer)
        return self

    def __next__(self):
        try:
            return self._sess.run(self.frame)
        except tf.errors.OutOfRangeError:
            raise StopIteration
