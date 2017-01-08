import numpy as np
import tensorflow as tf
from collections import OrderedDict
from utils import Struct

'''
Utilities to construct neural nets and keep track of intermediate computations
The whole kwargs passing is currently a mess - better to avoid using
'''


def Glorot(shape, scale=1.):
    scale = scale * np.sqrt(6.0 / np.sum(shape))
    return tf.Variable(tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=np.float32))


def Glorot_np(shape, scale=1.):
    scale = scale * np.sqrt(6.0 / np.sum(shape))
    return tf.Variable((np.random.rand(*shape) * 2 * scale - scale).astype(np.float32))


def ZeroBias(len):
    return tf.Variable(np.zeros((len,), dtype=np.float32))


def leaky_relu(x):
    return tf.maximum(0.01 * x, x)


def bndsoftplus(x):
    return tf.nn.softplus(tf.minimum(tf.maximum(x, -10.), 10.))


class NL():
    def __init__(self, nl):
        self.nl = nl
        self.params = []
        self.cache = {}

    def __call__(self, x, **kwargs):
        if x not in self.cache:
            y = self.nl(x)
            self.cache[x] = Struct(y=y)
        return self.cache[x]

    def to_short_str(self):
        to_str = {tf.nn.relu: 'relu',
                  tf.nn.sigmoid: 'sigm',
                  tf.tanh: 'tanh',
                  leaky_relu: 'lrelu',
                  tf.nn.softplus: 'softpl',
                  bndsoftplus: 'softplbd'}
        return to_str[self.nl]

Tanh = NL(tf.tanh)
Relu = NL(tf.nn.relu)
Sigmoid = NL(tf.nn.sigmoid)
LeakyRelu = NL(leaky_relu)
Softplus = NL(tf.nn.softplus)
BndSoftplus = NL(bndsoftplus)


def scaled_tanh(min_sc, max_sc):
    # to be used as 
    # ScaledTanh = NL(scaled_tanh(-1.1, 1.2))
    return lambda x: tf.nn.sigmoid(x) * (max_sc - min_sc) + min_sc


class Const():
    def __init__(self, c, trainable=True):
        self.params = [c] if trainable else []
        self.c = c if trainable else tf.stop_gradient(c)
        self.cache = {}

    def __call__(self, n, **kwargs):
        if n not in self.cache:
            c_rep = tf.zeros(tf.pack([n, 1])) + self.c
            self.cache[n] = Struct(y=c_rep)
        return self.cache[n]


class Affine():
    def __init__(self, W, b, nonlinearity=None):
        '''
        Inputs and outputs might not be specified in the beginning
        '''
        self.W = W
        self.b = b
        self.params = [W, b]
        self.cache = {}

    def __call__(self, x, **kwargs):
        if x not in self.cache:
            xW = tf.matmul(x, self.W)
            y = xW if self.b is None else xW + self.b
            self.cache[x] = Struct(xW=xW, y=y)
        return self.cache[x]

    def to_short_str(self):
        return 'lin{}'.format(self.W.get_shape().as_list()[1])

    @staticmethod
    def build_affine(n1, n2):
        return Affine(W=Glorot(shape=(n1, n2), scale=1.), b=ZeroBias(n2))


class Dropout():
    def __init__(self, keep_rate):
        self.keep_rate = keep_rate
        self.params = []
        self.cache = {}

    def __call__(self, x, deterministic=False, **kwargs):
        if (x, deterministic) not in self.cache:
            if deterministic:
                mask = None
                y = x
            else:
                mask = tf.cast(tf.less_equal(tf.random_uniform(tf.shape(x)), self.keep_rate), np.float32)
                y = x * self.mask / self.keep_rate
            self.cache[x, deterministic] = Struct(mask=mask, y=y)
        return self.cache[x, deterministic]

    def to_short_str(self):
        return 'dp{}'.format(self.keep_rate)


class BitFlip():
    def __init__(self, keep_rate):
        self.keep_rate = keep_rate
        self.params = []
        self.cache = {}

    def __call__(self, x, deterministic=False, **kwrags):
        if (x, deterministic) not in self.cache:
            if deterministic:
                mask = None
                y = x * self.keep_rate + (1. - x) * (1. - self.keep_rate)
            else:
                mask = tf.cast(tf.less_equal(tf.random_uniform(tf.shape(x)), self.keep_rate), np.float32)
                y = x * mask + (1. - x) * (1. - mask)
            self.cache[x, deterministic] = Struct(mask=mask, y=y)
        return self.cache[x]

    def to_short_str(self):
        return 'bf{}'.format(self.keep_rate)


class BatchNorm():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.params = [mu, sigma]
        self.cache = {}

    def __call__(self, x, **kwargs):
        if x not in self.cache:
            m, v = tf.nn.moments(x, [0])
            x_cen = (x - m) / (tf.sqrt(v) + 1e-4)
            y = x_cen * self.sigma + self.mu
            self.cache[x] = Struct(m=m, v=v, x_cen=x_cen, y=y)
        return self.cache[x]

    @staticmethod
    def build(n):
        mu = tf.Variable(np.zeros(n, dtype=np.float32))
        sigma = tf.Variable(np.ones(n, dtype=np.float32))
        return BatchNorm(mu, sigma)

    def to_short_str(self):
        return 'bn'


class Identity():
    def __init__(self):
        self.params = []
        self.cache = {}

    def __call__(self, x, **kwargs):
        if x not in self.cache:
            self.cache[x] = Struct(y=x)
        return self.cache[x]


class Split():
    def __init__(self, *functions):
        self.functions = functions
        params = sum([f.params for f in self.functions], [])
        # remove duplicate parameters if any, while preserving the order
        self.params = list(OrderedDict.fromkeys(params))
        self.cache = {}

    def __call__(self, x, list_of_kwargs=None, **kwargs):
        if list_of_kwargs is None:
            list_of_kwargs = []
        while len(list_of_kwargs) < len(self.functions):
            list_of_kwargs.append({})
        assert len(list_of_kwargs) == len(self.functions), "more arguments than splits"
        if x not in self.cache:
            outs = []
            for f, kwargs in zip(self.functions, list_of_kwargs):
                outs.append(f(x, **kwargs))
            self.cache[x] = Struct(outs=outs, y=tuple([out.y for out in outs]))
        return self.cache[x]


class PlusMerge():
    def __init__(self):
        self.params = []
        self.cache = {}

    def __call__(self, xs):
        if xs not in self.cache:
            y = 0
            for x in xs:
                y += x
            self.cache[xs] = Struct(y=y)
        return self.cache[xs]


class InterpolateMerge():
    def __init__(self, alpha, trainable=False):
        self.alpha = alpha if trainable else tf.stop_gradient(alpha)
        self.params = [] if not trainable else [self.alpha]
        self.cache = {}

    def __call__(self, xs, **kwargs):
        if xs not in self.cache:
            y = self.alpha * xs[0] + (1. - self.alpha) * xs[1]
            self.cache[xs] = Struct(y=y)
        return self.cache[xs]


class Chain():
    def __init__(self, *layers):
        self.layers = layers
        params = sum([layer.params for layer in self.layers], [])
        # remove duplicate parameters if any, while preserving the order
        self.params = list(OrderedDict.fromkeys(params))
        self.cache = {}

    def __call__(self, x, deterministic=False, **kwrags):
        if (x, deterministic) not in self.cache:
            intermediate_outs = OrderedDict()
            y = x
            for layer in self.layers:
                out = layer(y, deterministic=deterministic, **kwrags)
                y = out.y
                intermediate_outs[layer] = out
            self.cache[x] = Struct(intermediate_outs=intermediate_outs, y=y, out=out)

        return self.cache[x]

    @staticmethod
    def build_chain(ns, nonlinearity, last_nonlinearity=None, dropout=False, last_b=None, batch_norm=False, batch_norm_alpha=1):
        def layer_maker(n1, n2):
            l = [Affine(W=Glorot(shape=(n1, n2), scale=1.), b=ZeroBias(n2))]
            if batch_norm:
                if batch_norm_alpha is 1:
                    l.append(BatchNorm.build(n2))
                else:
                    l.append(
                        Split(Identity(), BatchNorm.build(n2))
                    )
                    l.append(InterpolateMerge(batch_norm_alpha))
            if nonlinearity is not None:
                l.append(nonlinearity)
            if dropout is not False:
                l.append(Dropout(dropout))
            return l
        layers = sum([layer_maker(n_in, n_out) for n_in, n_out in zip(ns[:-1], ns[1:-1])], [])
        if last_b is None:
            b = ZeroBias(ns[-1])
        else:
            b = tf.Variable(last_b.astype(np.float32))
        layers.append(Affine(W=Glorot(shape=(ns[-2], ns[-1]), scale=1.), b=b))
        if last_nonlinearity is not None:
            layers.append(nonlinearity)
        return Chain(*layers)


def residual_block(layers):
    l = []
    l.append(Split(Identity(), Chain(layers)))
    l.append(PlusMerge())
    return Chain(l)


# class MakeZeroMeanUnitVariance():
#     def __init__(self, chain):
#         self.chain = chain.clear_copy()

#     def __call__(self, x):
#         self.corrected_ys = []
#         dependencies = []
#         for layer in self.chain.layers:
#             y = layer(x)
#             if isinstance(layer, Affine):
#                 mean, var = tf.nn.moments(layer.xW, [0], keep_dims=True)
#                 std = tf.sqrt(var)
#                 correct_b = tf.assign(layer.b, -mean[0, :]/std[0, :])
#                 correct_W = tf.assign(layer.W, layer.W/std)
#                 dependencies = [correct_b, correct_W]
#                 with tf.control_dependencies(dependencies):  # recompute the output
#                     y = layer(x)
#                     self.corrected_ys.append(y)
#             x = y
#         return x

