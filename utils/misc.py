import numpy as np
import progressbar
import urllib
import tensorflow as tf


class Struct():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, other):
        if isinstance(other, dict):
            self.__dict__.update(other)
        elif isinstance(other, Struct):
            self.__dict__.update(other.__dict__)

    def updated(self, other):
        copy = Struct(**self.__dict__)
        copy.update(other)
        return copy

    def update_exclusive(self, other):
        if isinstance(other, dict):
            d = other
        elif isinstance(other, Struct):
            d = other.__dict__
        for x in d:
            if x not in self.__dict__:
                self.__dict__[x] = d[x]
        return self

    def __repr__(self):
        return ''.join(['Struct('] + ['{}: {}\n'.format(repr(x), repr(y)) for x, y in self.__dict__.iteritems()]+[')'])


def binarize(x, rng):
    return (rng.rand(*x.shape) < x).astype(np.float32)


def download(url, filename):
    print "Downloading"
    pbar = progressbar.ProgressBar()

    def dlProgress(count, blockSize, totalSize):
        if pbar.maxval is None:
            pbar.maxval = totalSize
            pbar.start()

        pbar.update(min(count*blockSize, totalSize))

    urllib.urlretrieve(url, filename, reporthook=dlProgress)
    pbar.finish()


def convert_to_grayscale(array):
    '''
    Equivalent to PIL's image.convert('L')
    array is an image in height x width x color format,
    color is rgb
    '''
    conversion = np.array([299, 587, 114])
    return array.dot(conversion)/1000


def tile_images(array, n_cols=None):
    if n_cols is None:
        n_cols = int(np.sqrt(array.shape[0]))
    n_rows = int(np.ceil(float(array.shape[0])/n_cols))

    def cell(i, j):
        ind = i*n_cols+j
        if i*n_cols+j < array.shape[0]:
            return array[ind]
        else:
            return np.zeros(array[0].shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)


def print_and_save(filename, *x):
        s = ' '.join([str(_) for _ in x])
        print s
        with open(filename, 'a') as f:
            f.write('{}{}'.format(s, '\n'))

def tf_repeat(x, k):
    # return tf.reshape(tf.tile(x, tf.pack([1, k])), [-1, x.get_shape().as_list()[1]])
    return tf.reshape(tf.tile(x, tf.pack([1, k])), tf.pack([-1, tf.shape(x)[1]]))

def tf_log_mean_exp(x):
    m = tf.reduce_max(x, 1, keep_dims=True)
    return m + tf.log(tf.reduce_mean(tf.exp(x - m), 1, keep_dims=True))
