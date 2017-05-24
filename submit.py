#!/usr/bin/env python
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import gzip


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
    f: A file object that can be passed into a gzip reader.
    Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
    ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)

        if magic != 2051:
            raise ValueError(
                'Error in MNIST image file: %s' % (magic, f.name))

        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
    Returns:
    labels: a 1D uint8 numpy array.
    Raises:
    ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)

        if magic != 2049:
            raise ValueError(
                'Error in MNIST label file: %s' % (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
    return labels


def read_datasets():
    train_images = None
    train_labels = None
    test_images = None
    test_labels = None

    with open("./Mnist_data/train-images-idx3-ubyte.gz", 'rb') as f:
        train_images = extract_images(f)

    with open("./Mnist_data/train-labels-idx1-ubyte.gz", 'rb') as f:
        train_labels = extract_labels(f)

    with open("./Mnist_data/t10k-images-idx3-ubyte.gz", 'rb') as f:
        test_images = extract_images(f)

    with open("./Mnist_data/t10k-labels-idx1-ubyte.gz", 'rb') as f:
        test_labels = extract_labels(f)

    return train_images, train_labels, test_images, test_labels


def train_valid_test(train_X, train_y, test_X, test_y):
    def softmax(x):
        try:
            x = x - np.max(x, axis=1)[:, np.newaxis]
            return np.exp(x) / np.sum(np.exp(x), axis=1)[:, np.newaxis]
        except:
            return np.exp(x) / np.sum(np.exp(x), axis=1)[:, np.newaxis]

    def cross_entropy(y, t):
        batch_size = y.shape[0]
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        if t.size == y.size:
            t = t.argmax(axis=1)

        eps = 0.001
        return -np.sum(np.log(y[np.arange(batch_size), t] + eps)) / batch_size

    def to_onehot(y):
        onehot = np.zeros((y.shape[0], 10))
        onehot[np.arange(y.shape[0]), y] = 1
        return onehot

    class Relu:
        def __init__(self):
            self.mask = None

        def forward(self, x):
            self.mask = (x <= 0)
            out = x.copy()
            out[self.mask] = 0
            return out

        def backward(self, dx):
            dx[self.mask] = 0
            dx = dx
            return dx

    class Fully_Layer:
        def __init__(self, input_dim, output_dim):
            self._w = np.random.randn(input_dim, output_dim) * 0.01
            self._b = np.zeros((output_dim))
            self._x = None
            self._lr = 0.01

        def forward(self, x):
            self._x = x
            return np.dot(self._x, self._w) + self._b

        def backward(self, dx, lr=0.01):
            self._lr = lr
            self._w -= self._lr * np.dot(self._x.T, dx)
            self._b -= self._lr * np.sum(dx, axis=0)
            return np.dot(dx, self._w.T)

    class SoftmaxWithLoss:
        def __init__(self):
            self.loss = None
            self.y = None
            self.t = None

        def forward(self, x, t):
            self.t = t
            self.y = softmax(x)
            self.loss = cross_entropy(self.y, self.t)
            return self.loss

        def backward(self):
            return (self.y - self.t) / self.t.shape[0]

    train_X, valid_X, train_y, valid_y = train_test_split(
        train_X,
        train_y,
        test_size=0.2,
        random_state=42
    )
    train_Y = to_onehot(train_y)

    layer_arch = {
        "layer1": 512,
        "layer2": 10,
    }

    layer1 = Fully_Layer(train_X.shape[1], layer_arch["layer1"])
    relu1 = Relu()
    layer2 = Fully_Layer(layer_arch["layer1"], layer_arch["layer2"])
    loss = SoftmaxWithLoss()

    batch_size = 32
    iter_num = int(np.ceil(train_X.shape[0] / batch_size))
    epoch = 50
    lr = 0.01

    for ep in range(epoch):
        if lr > 0.001:
            lr = lr * 0.9
        for it in range(iter_num):
            output = layer1.forward(
                train_X[it * batch_size:(it + 1) * batch_size])
            output = relu1.forward(output)
            output = layer2.forward(output)
            output = loss.forward(
                output, train_Y[it * batch_size:(it + 1) * batch_size])
            dx = loss.backward()
            dx = layer2.backward(dx, lr=lr)
            dx = relu1.backward(dx)
            dx = layer1.backward(dx, lr=lr)

        if ep % 2 == 0:
            pred_y = np.argmax(layer2.forward(
                relu1.forward(layer1.forward(valid_X))), axis=1)
            print("epoch {} valid f1 score: {}".format(ep,
                f1_score(valid_y, pred_y, average='macro')))

    pred_y = np.argmax(layer2.forward(
        relu1.forward(layer1.forward(test_X))), axis=1)
    return pred_y


def load_mnist():
    train_X, train_y, test_X, test_y = read_datasets()
    train_X, train_y = shuffle(train_X, train_y, random_state=42)
    test_X, test_y = shuffle(test_X, test_y, random_state=42)
    return train_X / 255., train_y, test_X / 255., test_y


def validate_model():
    print("######## Validate model ########")
    train_X, train_y, test_X, test_y = load_mnist()
    train_X = train_X.reshape((train_X.shape[0], -1))
    test_X = test_X.reshape((test_X.shape[0], -1))

    # validate for small dataset
    train_X_mini = train_X[:5000]
    train_y_mini = train_y[:5000]
    test_X_mini = test_X[:100]
    test_y_mini = test_y[:100]

    pred_y = train_valid_test(
        train_X_mini, train_y_mini, test_X_mini, test_y_mini)
    accuracy = pred_y[pred_y == test_y_mini].shape[0] / float(pred_y.shape[0])
    print("accuracy: {}".format(accuracy))


def score_model():
    print("######## Score model ########")
    train_X, train_y, test_X, test_y = load_mnist()
    train_X = train_X.reshape((train_X.shape[0], -1))
    test_X = test_X.reshape((test_X.shape[0], -1))

    pred_y = train_valid_test(train_X, train_y, test_X, test_y)
    accuracy = pred_y[pred_y == test_y].shape[0] / float(pred_y.shape[0])
    print("accuracy: {}".format(accuracy))


if __name__ == '__main__':
    validate_model()
    score_model()
