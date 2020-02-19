from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical


class Data(object):
  def __init__(self, x_train, x_test, y_train, y_test):
    self.train = x_train
    self.test = x_test
    self.y_train = y_train
    self.y_test = y_test

  @staticmethod
  def mnist(fashion=False):
    (x_train, y_train), (x_test, y_test) = (
      fashion_mnist.load_data() if fashion else mnist.load_data())

    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    return Data(x_train, x_test, y_train, y_test)

  @staticmethod
  def mnist_bin(fashion=False):
    (x_train, y_train), (x_test, y_test) = (
      fashion_mnist.load_data() if fashion else mnist.load_data())

    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Binarize to 1s and 7s as in Jordan et al.
    x_train = x_train[(y_train == 1) + (y_train == 7)]
    x_test = x_test[(y_test == 1) + (y_test == 7)]

    y_train = y_train[(y_train == 1) + (y_train == 7)]
    y_train[y_train == 1] = 0
    y_train[y_train == 7] = 1

    y_test = y_test[(y_test == 1) + (y_test == 7)]
    y_test[y_test == 1] = 0
    y_test[y_test == 7] = 1

    return Data(x_train, x_test, y_train, y_test)
