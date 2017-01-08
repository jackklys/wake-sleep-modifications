import tensorflow as tf
import numpy as np
import datasets
from utils import config, Struct
import utils
import os
import sys
from nnet import Affine, Chain, LeakyRelu, BernoulliLogits, Const, ZeroBias
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Model():
    def __init__(self, x, latent_units, q_units, p_units, sleep='usual', batch_norm=True):
        self.x = x
        self.mb_size = tf.shape(x)[0]
        self.latent_units = latent_units
        datadim = self.x.get_shape().as_list()[1]
        self.q_units = []
        for n1, n2, u in zip([datadim] + latent_units, latent_units, q_units):
            self.q_units.append([n1] + u + [n2])
        self.p_units = []
        for n1, n2, u in zip(reversed(latent_units), list(reversed(latent_units))[1:] + [datadim], p_units):
            self.p_units.append([n1] + u + [n2])
        self._setup_connectivity(batch_norm)
        self._wake_phase()
        self._sleep_phase(sleep)

        self.loss = - (self.log_q_sleep + self.log_p_wake)

        self._test_loss()

    def _setup_connectivity(self, batch_norm):
        self.alpha = tf.Variable(0.)
        self.q = []
        for units in self.q_units:
            self.q.append(
                Chain(Chain.build_chain(units, LeakyRelu, batch_norm=batch_norm, batch_norm_alpha=self.alpha),
                      BernoulliLogits()
                      )
            )
        self.p = []
        for units in self.p_units:
            self.p.append(
                Chain(Chain.build_chain(units, LeakyRelu, batch_norm=batch_norm, batch_norm_alpha=self.alpha),
                      BernoulliLogits()
                      )
            )
        self.prior = Chain(Const(ZeroBias(self.latent_units[-1])),
                           BernoulliLogits()
                           )

    def _wake_phase(self):
        self.q_samples = [self.x]
        for q in self.q:
            self.q_samples.append(tf.stop_gradient(q(self.q_samples[-1]).out.sample))
        self.log_p_wake = self.prior(self.mb_size).out.log_prob(self.q_samples[-1])
        for p, sample1, sample2 in zip(reversed(self.p), self.q_samples[1:], self.q_samples):
            self.log_p_wake += p(sample1).out.log_prob(sample2)

    def _sleep_phase(self, sleep_kind):
        # if not (sleep_kind == 'top_layer_mix' or sleep_kind == 'top_down_mix'):
        self.prior_samples = [self.prior(self.mb_size).out.sample]
        for p in self.p:
            self.prior_samples.append(p(self.prior_samples[-1]).out.sample)

        # this is the crucial part
        if sleep_kind == 'usual':
            self.sleep_samples = self.prior_samples
        elif sleep_kind == 'autoenc':
            self.sleep_samples = list(reversed(self.q_samples))
        elif sleep_kind == 'mix':
            # when alpha is zero, take the sample from q; when 1 - from the prior
            self.mask = tf.cast(tf.random_uniform([1]) < self.alpha, tf.float32)
            self.sleep_samples = [self.mask * prior_sample + (1 - self.mask) * q_sample
                                  for prior_sample, q_sample in zip(self.prior_samples, reversed(self.q_samples))]
        elif sleep_kind == 'mixy_mix':
            # when alpha is zero, take the sample from q; when 1 - from the prior
            self.mask = tf.cast(tf.random_uniform([len(self.latent_units)+1,1]) < self.alpha, tf.float32)
            self.sleep_samples = [mask * prior_sample + (1 - mask) * q_sample
                                  for mask, prior_sample, q_sample in zip(self.mask, self.prior_samples, reversed(self.q_samples))]
        elif sleep_kind == 'reconstruct':
            # p samples sleep_samples from last q sample instead of prior sample
            self.sleep_samples = [self.q_samples[-1]]
            for p in self.p:
                self.sleep_samples.append(p(self.sleep_samples[-1]).out.sample)
        elif sleep_kind == 'top_down_mix1':
            # sample sleep from q sample at floor(alpha*10) layer
            layer = len(self.latent_units) - tf.floor((len(self.latent_units)+1)*self.alpha)
            i = tf.constant(0.)
            self.mask = tf.cast((layer < i),tf.float32)
            self.sleep_samples = [(1-self.mask) * self.q_samples[-1] + self.mask * self.prior_samples[0]]
            for q_sample, p in zip(reversed(self.q_samples[:-1]), self.p):
                self.sleep_samples.append( self.mask * p(self.sleep_samples[-1]).out.sample + (1-self.mask) * q_sample)
                i = i+1
                self.mask = tf.cast((layer < i),tf.float32)
        elif sleep_kind == 'top_down_mix2':
            # use alpha to determine prior sample or q samples. if q samples then sample sleep from q sample at floor(alpha*10) layer
            if tf.cast(tf.random_uniform([1]) < self.alpha, tf.int32)==1.: self.sleep_samples = self.prior_samples
            else:
                layer = len(self.latent_units) - tf.floor((len(self.latent_units)+1)*self.alpha)
                i = tf.constant(0.)
                self.mask = tf.cast((layer < i),tf.float32)
                self.sleep_samples = [(1-self.mask) * self.q_samples[-1] + self.mask * self.prior_samples[0]]
                for q_sample, p in zip(reversed(self.q_samples[:-1]), self.p):
                    self.sleep_samples.append( self.mask * p(self.sleep_samples[-1]).out.sample + (1-self.mask) * q_sample)
                    i = i+1
                    self.mask = tf.cast((layer < i),tf.float32)

        self.log_q_sleep = 0
        for q, p, sample in zip(reversed(self.q), self.p, self.sleep_samples):
            sample1 = tf.stop_gradient(p(sample).out.sample)
            self.log_q_sleep += q(sample1).out.log_prob(sample)

    def _test_loss(self):
        self.k = tf.placeholder(tf.int32)
        self.x_rep = utils.tf_repeat(self.x, self.k)
        self.q_samples_rep = [self.x_rep]
        for q in self.q:
            self.q_samples_rep.append(q(self.q_samples_rep[-1]).out.sample)

        self.variational_lower_bound = self.prior(self.k*self.mb_size).out.log_prob(self.q_samples_rep[-1])
        for p, sample1, sample2 in zip(reversed(self.p), self.q_samples_rep[1:], self.q_samples_rep):
            self.variational_lower_bound += p(sample1).out.log_prob(sample2)
        for q, sample1, sample2 in zip(self.q, self.q_samples_rep, self.q_samples_rep[1:]):
            self.variational_lower_bound -= q(sample1).out.log_prob(sample2)

        self.variational_lower_bound = tf.reshape(self.variational_lower_bound, tf.pack([-1, self.k]))
        self.variational_lower_bound = utils.tf_log_mean_exp(self.variational_lower_bound)

        self.reconstruction_loss = self.p[-1](self.q_samples_rep[1]).out.log_prob(self.x_rep)
        self.reconstruction_loss = tf.reshape(self.reconstruction_loss, tf.pack([-1, self.k]))
        self.reconstruction_loss = tf.reduce_mean(self.reconstruction_loss, 1)


def main(args):
    args.update_exclusive(default_args)
    directory = os.path.join(config.RESULTSDIR, 'ws', args.directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not args.restore:
        map(os.remove, (os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt") or f.endswith(".png")))
        with open(os.path.join(directory, 'params.txt'), 'w') as f:
            f.write(repr(args))
    print directory

    dataset = datasets.MNIST(binary=True)

    x = tf.placeholder(np.float32, shape=(None, dataset.get_data_dim()))

    model = Model(x, args.latent_units,
                  q_units=args.q_units,
                  p_units=args.p_units,
                  sleep=args.sleep_type,
                  batch_norm=args.bn)
    examples_per_epoch = dataset.data['train'][0].shape[0]
    num_updates = args.n_epochs * examples_per_epoch / args.mb_size

    step = tf.Variable(0, trainable=False)
    lr = tf.placeholder(tf.float32)
    train_op = args.optimizer(lr).minimize(model.loss, global_step=step)
    with tf.control_dependencies([train_op]):
        # linearly anneal alpha from 0 to 1 over the course of N/2 epochs, then train for additional N/2 epochs with alpha=1
        train_op = tf.assign(model.alpha, tf.minimum(1., tf.maximum(0., tf.cast(step, tf.float32) / num_updates * 2)))

    init_op = tf.initialize_all_variables()


    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        if not args.restore:
            sess.run(init_op)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join(directory)))
            print "restored"
            measure_test_log_likelihood(sess, model, dataset, directory)
            import ipdb; ipdb.set_trace()
        for x_np, _ in dataset.random_minibatches('train', args.mb_size, num_updates):
            i, _ = sess.run([step, train_op], feed_dict={x: x_np, lr: args.lr})
            if i % 1000 == 1 or i == num_updates - 1:
                visualize(sess, model, dataset, directory, float(i) * args.mb_size / examples_per_epoch)
            if i % 10000 == 1 or i == num_updates - 1:
                saver.save(sess, os.path.join(directory, 'model.chk'), global_step=step)
            if i % 10000 == 1:
                print directory


def visualize(sess, model, dataset, directory, epoch):
    x_np, _ = dataset.get_random_minibatch('train', 100, np.random.RandomState(123))
    x_np_valid, _ = dataset.get_random_minibatch('valid', 100, np.random.RandomState(123))
    q_means = [q(s).out.mu for q, s in zip(model.q, model.q_samples)]
    q_means = sess.run(q_means, feed_dict={model.x: x_np})
    for i, q_m in enumerate(q_means):
        plt.boxplot(q_m[:, np.argsort(q_m.std(axis=0))])
        # plt.violinplot(q_means)
        plt.savefig(os.path.join(directory, 'means of q{}.png'.format(i)))
        plt.close()
    prior_means = sess.run(model.prior(1).out.mu)[0]
    plt.scatter(np.arange(prior_means.shape[0]), prior_means)
    plt.savefig(os.path.join(directory, 'prior means.png'))
    plt.close()
    # pre_activations = [layer.y for layer in model.q_params_chain.all_layers]
    # names = [layer.to_short_str() for layer in model.q_params_chain.all_layers]
    # pre_activations = sess.run(pre_activations, feed_dict={model.x: x_np})
    # for i, (act, name) in enumerate(zip(pre_activations, names)):
    #     plt.violinplot(act[:, np.argsort(act.std(axis=0))])
    #     plt.savefig(os.path.join(directory, 'rec_activations{}{}.png'.format(i+1, name)))
    #     plt.close()
    # pre_activations = [layer.y for layer in model.p_samples_copy.layers]
    # names = [layer.to_short_str() for layer in model.p_samples_copy.layers]
    # pre_activations = sess.run(pre_activations, feed_dict={model.num_samples: 100})
    # for i, (act, name) in enumerate(zip(pre_activations, names)):
    #     plt.violinplot(act[:, np.argsort(act.mean(axis=0))])
    #     plt.savefig(os.path.join(directory, 'gen_model_activations{}{}.png'.format(i+1, name)))
    #     plt.close()
    first_linear_layer = next(layer for layer in model.q[0].layers[0].layers if isinstance(layer, Affine))
    order = np.argsort(sess.run(model.q[0].layers[0].layers[0](model.x).y, feed_dict={model.x: x_np}).std(axis=0))
    W = sess.run(first_linear_layer.W)
    plt.imshow(utils.misc.tile_images(dataset.reshape_for_display(W.T[order])), cmap='gray')
    plt.savefig(os.path.join(directory, 'rec_weights.png'))
    plt.close()
    last_linear_layer = [layer for layer in model.p[-1].layers[0].layers if isinstance(layer, Affine)][-1]
    W = sess.run(last_linear_layer.W)
    plt.imshow(utils.misc.tile_images(dataset.reshape_for_display(W)), cmap='gray')
    plt.savefig(os.path.join(directory, 'gen_weights.png'))
    plt.close()
    samples, recs = sess.run([model.p[-1](model.prior_samples[-2]).out.mu,
                              model.p[-1](model.q[0](model.x).out.sample).out.mu], feed_dict={model.x: x_np})
    samples = utils.misc.tile_images(dataset.reshape_for_display(samples))
    recs = utils.misc.tile_images(dataset.reshape_for_display(recs))
    orig = utils.misc.tile_images(dataset.reshape_for_display(x_np))
    orig_and_samples = np.concatenate([orig, recs, samples], axis=1)
    plt.imshow(orig_and_samples, cmap='gray')
    plt.savefig(os.path.join(directory, 'samples.png'))
    plt.close()

    loss_1 = sess.run(model.variational_lower_bound, feed_dict={model.x: x_np, model.k: 1})
    loss_10 = sess.run(model.variational_lower_bound, feed_dict={model.x: x_np, model.k: 10})
    loss_100, rec_100 = sess.run([model.variational_lower_bound, model.reconstruction_loss],
                                 feed_dict={model.x: x_np, model.k: 100})
    loss_valid, rec_valid = sess.run([model.variational_lower_bound, model.reconstruction_loss],
                                     feed_dict={model.x: x_np_valid, model.k: 100})
    utils.print_and_save(os.path.join(directory, 'loss.txt'),
                         'epoch ', epoch,
                         'train rec100', np.mean(rec_100),
                         'valid rec100', np.mean(rec_valid),
                         'train loss1', np.mean(loss_1),
                         'train loss10', np.mean(loss_10),
                         'train loss100', np.mean(loss_100),
                         'valid loss100', np.mean(loss_valid),
                         'alpha', sess.run(model.alpha))

    # loss = measure_validation_loss(sess, model, dataset)
    # utils.print_and_save(os.path.join(directory, 'loss.txt'), 'train', cum_loss, 'valid', loss)


def measure_test_log_likelihood(sess, model, dataset, directory, k=5000):
    mbsize = 10
    ll = 0.
    n = 0
    for x_np, _ in dataset.all_minibatches('test', mbsize, np.random.RandomState(123)):
        n += x_np.shape[0]
        ll += np.sum(sess.run(model.variational_lower_bound, feed_dict={model.x: x_np, model.k: k}))
        sys.stdout.write(str(n)+','+str(ll/n))
    utils.print_and_save(os.path.join(directory, 'loss.txt'),
                         'test log likelihood 5000', ll/n)


default_args = Struct(
    optimizer=tf.train.AdamOptimizer,
    sleep_type='usual',
    mb_size=200,
    latent_units=50,
    q_units=[500, 500],
    p_units=[500, 500],
    bn=True,
    lr=0.001,
    n_epochs=500,
    directory='default',
    restore=False
    )
