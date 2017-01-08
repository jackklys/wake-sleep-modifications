import numpy as np
import tensorflow as tf
from utils import Struct

log_2pi = np.log(2 * np.pi)


# class UnitUniform():
#     def __init__(self):
#         pass

#     def __call__(self, shape):
#         self.noise = tf.random_uniform(shape)
#         self.sample = self.noise
#         return self

#     def prob(self, x):
#         return tf.cast(tf.reduce_all(tf.logical_and(tf.less(x, 1.), tf.greater_equal(x, 0.)), 1), np.float32)

#     def clear_copy(self):
#         return UnitUniform()


class DiagonalGaussian():
    class DiagonalGaussianLogProb():
        def __init__(self, mu, sigma):
            self.params = []
            self.mu = mu
            self.sigma = sigma
            self.dim = tf.shape(mu)[1]
            self.cache = {}

        def __call__(self, x):
            if x not in self.cache:
                norm_x = (x - self.mu) / self.sigma
                log_prob = - log_2pi / 2 * tf.cast(self.dim, tf.float32) - tf.reduce_sum(tf.log(self.sigma), 1) \
                    - tf.reduce_sum(tf.square(norm_x), 1) / 2
                self.cache[x] = log_prob
            return self.cache[x]

    def __init__(self):
        self.params = []
        self.cache = {}

    def __call__(self, mu_sigma, **kwargs):
        if mu_sigma not in self.cache:
            mu, sigma = mu_sigma
            noise = tf.random_normal(tf.shape(mu))
            sample = mu + sigma * noise
            log_prob = DiagonalGaussian.DiagonalGaussianLogProb(mu, sigma)
            self.cache[mu_sigma] = Struct(mu=mu, sigma=sigma, noise=noise, y=sample, sample=sample, log_prob=log_prob)
        return self.cache[mu_sigma]

    def clear_copy(self):
        return DiagonalGaussian()


# class UnitGaussian():
#     def __init__(self):
#         pass

#     def __call__(self, shape):
#         self.sample = tf.random_normal(shape)
#         return self

#     def log_prob(self, x):
#         return - log_2pi / 2 * x.get_shape().as_list()[1] \
#             - tf.reduce_sum(tf.square(x), 1) / 2

#     def clear_copy(self):
#         return UnitGaussian()


# class Bernoulli():
#     def __init__(self):
#         pass

#     def __call__(self, mu):
#         self.mu = mu
#         self.dim = self.mu.get_shape().as_list()[1]
#         self.noise = tf.random_uniform(tf.shape(self.mu))
#         self.sample = tf.cast(tf.less_equal(self.noise, self.mu), np.float32)

#     def log_prob(self, x):
#         return tf.reduce_sum(x * tf.log(self.mu) + (1-x)*tf.log(1-self.mu), 1)

#     def clear_copy(self):
#         return Bernoulli()


class BernoulliLogits():
    class BernoulliLogitsLogProb():
        def __init__(self, logits):
            self.logits = logits
            self.cache = {}

        def __call__(self, x):
            if x not in self.cache:
                log_prob = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, x), 1)
                self.cache[x] = log_prob
            return self.cache[x]

    def __init__(self):
        self.params = []
        self.cache = {}

    def __call__(self, logits, **kwargs):
        if logits not in self.cache:
            mu = tf.sigmoid(logits)
            noise = tf.random_uniform(tf.shape(logits))
            sample = tf.cast(tf.less_equal(noise, mu), np.float32)
            log_prob = BernoulliLogits.BernoulliLogitsLogProb(logits)
            self.cache[logits] = Struct(logits=logits, mu=mu, noise=noise, y=sample, sample=sample, log_prob=log_prob)
        return self.cache[logits]
