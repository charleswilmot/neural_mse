import re
import tensorflow as tf


def lrelu(x, name=None):
    alpha = 1 / 3
    return alpha * x + (1 - alpha) * tf.nn.relu(x, name=name)


def str_to_activation(string):
    if string == 'relu':
        return tf.nn.relu
    elif string == 'lrelu':
        return lrelu
    elif string == 'sig':
        return tf.sigmoid
    elif string == 'tanh':
        return tf.tanh
    elif string == 'softmax':
        return tf.nn.softmax
    elif string == 'none':
        return tf.identity
    else:
        raise ValueError("Activation function does not match (got {})".format(string))


class Network:
    def __init__(self, keys, input_shape):
        self._keys = keys
        self._regex = {
            'flat': re.compile(r'flatten'),
            'reshape': re.compile(r'reshape_(?:[0-9]+)+'),
            'conv': re.compile(r'conv_outdim([0-9]+)_size([0-9]+)_stride([0-9]+)_act_(relu|lrelu|sig|tanh|softmax|none)'),
            'deconv': re.compile(r'deconv_outdim([0-9]+)_size([0-9]+)_stride([0-9]+)_act_(relu|lrelu|sig|tanh|softmax|none)'),
            'fc': re.compile(r'fc_outdim([0-9]+)_act_(relu|lrelu|sig|tanh|softmax|none)')
        }
        self.model = tf.keras.Sequential()
        for key in self._keys:
            self._make(key)
        self.model.build(input_shape)

    def _make_flat(self, key):
        match = self._regex["flat"].match(key)
        if match is not None:
            print("Key\t\t{}".format(key))
            self.model.add(tf.keras.layers.Flatten())
            print("\n")
            return True
        return False

    def _make_reshape(self, key):
        match = self._regex["reshape"].match(key)
        if match is not None:
            match = re.findall(r'[0-9]+', key)
            print("Key\t\t{}".format(key))
            shape = [int(x) for x in match]
            print('shape\t\t{}'.format(shape))
            self.model.add(tf.keras.layers.Reshape(tuple(shape)))
            print("\n")
            return True
        return False

    def _make_conv(self, key):
        match = self._regex["conv"].match(key)
        if match is not None:
            print("Key\t\t{}".format(key))
            filters = int(match.group(1))
            kernel_size = int(match.group(2))
            strides = int(match.group(3))
            string = match.group(4)
            activation = str_to_activation(string)
            print('filters\t\t{}'.format(filters))
            print('k_size\t\t{}'.format(kernel_size))
            print('strides\t\t{}'.format(strides))
            print('act\t\t{}'.format(string))
            self.model.add(tf.keras.layers.Conv2D(filters, kernel_size, strides, activation=activation, padding='same'))
            print("\n")
            return True
        return False

    def _make_deconv(self, key):
        match = self._regex["deconv"].match(key)
        if match is not None:
            print("Key\t\t{}".format(key))
            filters = int(match.group(1))
            kernel_size = int(match.group(2))
            strides = int(match.group(3))
            string = match.group(4)
            activation = str_to_activation(string)
            print('filters\t\t{}'.format(filters))
            print('k_size\t\t{}'.format(kernel_size))
            print('strides\t\t{}'.format(strides))
            print('act\t\t{}'.format(string))
            self.model.add(tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, activation=activation, padding='same'))
            print("\n")
            return True
        return False

    def _make_fc(self, key):
        match = self._regex["fc"].match(key)
        if match is not None:
            print("Key\t\t{}".format(key))
            units = int(match.group(1))
            string = match.group(2)
            activation = str_to_activation(string)
            print('units\t\t{}'.format(units))
            print('act\t\t{}'.format(string))
            self.model.add(tf.keras.layers.Dense(units, activation=activation))
            print("\n")
            return True
        return False

    def _make(self, key):
        done = False
        done = done or self._make_flat(key)
        done = done or self._make_reshape(key)
        done = done or self._make_conv(key)
        done = done or self._make_deconv(key)
        done = done or self._make_fc(key)
        if not done:
            raise ValueError("the key does not match (got {})".format(key))

    def __call__(self, inp):
        return self.model(inp)

    def _get_variables(self):
        return self.model.variables

    variables = property(_get_variables)


def save_keys(path, keys):
    with open(path, 'w') as f:
        for k in keys:
            f.write(k + "\n")

def load_keys(path):
    with open(path, "r") as f:
        keys = f.readlines()
    return [x.strip() for x in keys]


if __name__ == '__main__':
    keys = [
        "conv_outdim32_size3_stride1_act_relu",
        "flatten",
        "fc_outdim20_act_relu",
        "fc_outdim20_act_relu",
        "fc_outdim20_act_relu",
        "fc_outdim20_act_none"
    ]
    input_shape = [None, 50, 50, 3]
    n = Network(keys, input_shape)
    inp = tf.placeholder(shape=(None, 50, 50, 3), dtype=tf.float32)
    out = n(inp)
    print('\n\n', out)
    print('\n\n', n.variables)
