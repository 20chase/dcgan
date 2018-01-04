import os
import time
import argparse

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from glob import glob
from random import shuffle
from utils import get_image


parser = argparse.ArgumentParser(description='DCGAN Example')

parser.add_argument(
    '--lr', default=5e-5, type=float, help='learning rate')

parser.add_argument(
    '--train_size', default=np.inf, type=float, help='training size')

parser.add_argument(
    '--g_update', default=2, type=int, help='generator update number per iter')

parser.add_argument(
    '--color_dim', default=3, type=int, help='the color channels')

parser.add_argument(
    '--epochs', default=50, type=int, help='training epochs')

parser.add_argument(
    '--filter_size', default=64, type=int, help='the conv filter size')

parser.add_argument(
    '--batch_size', default=64, type=int, help='batch size')

parser.add_argument(
    '--sample_size', default=64, type=int, help='sample_size')

parser.add_argument(
    '--image_size', default=64, type=int, help='image size')

parser.add_argument(
    '--z_dim', default=128, type=int, help='generator input dim')

parser.add_argument(
    '--sample_step', default=500, type=int, help='sample steps')

parser.add_argument(
    '--is_crop', default=True, type=bool, help='whether to crop image')

parser.add_argument(
    '--dataset', default='celebA', type=str, help='the name of dataset')

parser.add_argument(
    '--sample_dir', default='samples', type=str, help='the name of sample_dir')

args = parser.parse_args()

class DCGAN(object):
    def __init__(self, args):
        self.time_step = 0
        graph = tf.get_default_graph()
        config = tf.ConfigProto()
        self.sess = tf.Session(graph=graph, config=config)
        self.args = args
        self._build_ph()

        self.g_net, self.g_logits = self._build_generator(is_train=True, reuse=False)
        self.d_net, self.d_logits = self._build_discriminator(self.g_net.outputs, is_train=True, reuse=False)
        self.d2_net, self.d2_logits = self._build_discriminator(self.r_input_ph, is_train=True, reuse=True)
        self.g2_net, self.g2_logits = self._build_generator(is_train=False, reuse=True)

        self.g_opt, self.d_opt, self.g_loss, self.d_loss = self._build_training()
        self._build_tensorboard()

        self.merge_all = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('../tb/{}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))),
            self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_ph(self):
        self.z_ph = tf.placeholder(tf.float32, [None, self.args.z_dim], 'z_ph')
        self.r_input_ph = tf.placeholder(tf.float32,
            [None, self.args.image_size, self.args.image_size, self.args.color_dim],
            'r_input_ph')

    def _build_generator(self, is_train=True, reuse=False):
        s1, s2, s4, s8, s16 = int(self.args.image_size), int(self.args.image_size/ 2), int(self.args.image_size / 4), int(self.args.image_size/ 8), int(self.args.image_size / 16)

        filter_size = self.args.filter_size
        batch_size = self.args.batch_size
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., .02)
        with tf.variable_scope('generator', reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net = tl.layers.InputLayer(self.z_ph, name='g_input')
            net = tl.layers.DenseLayer(net, n_units=filter_size*8*s16*s16,
                W_init=w_init, name='g_linear')
            net = tl.layers.ReshapeLayer(net, shape=[-1, s16, s16, filter_size*8],
                name='g_reshape')
            net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_bn1')
            net = tl.layers.DeConv2d(net, filter_size*4, (3, 3), out_size=(s8, s8),
                strides=(2, 2), batch_size=batch_size, W_init=w_init, name='g_deconv1')
            net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_bn2')
            net = tl.layers.DeConv2d(net, filter_size*2, (3, 3), out_size=(s4, s4),
                strides=(2, 2), batch_size=batch_size, W_init=w_init, name='g_deconv2')
            net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_bn3')
            net = tl.layers.DeConv2d(net, filter_size, (3, 3), out_size=(s2, s2),
                strides=(2, 2), batch_size=batch_size, W_init=w_init, name='g_deconv3')
            net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_bn4')
            net = tl.layers.DeConv2d(net, self.args.color_dim, (3, 3), out_size=(s1, s1),
                strides=(2, 2), batch_size=batch_size, W_init=w_init, name='g_deconv4')

            logits = net.outputs
            net.outputs = tf.nn.tanh(net.outputs)

        return net, logits

    def _build_discriminator(self, inputs, is_train=True, reuse=False):
        filter_size = self.args.filter_size
        batch_size = self.args.batch_size
        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init = tf.random_normal_initializer(1., .02)
        with tf.variable_scope('discriminator', reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net = tl.layers.InputLayer(inputs, name='d_input')
            net = tl.layers.Conv2d(net, filter_size, (3, 3), (2, 2), act=lambda x: tl.act.lrelu(x, 0.3),
                W_init=w_init, name='d_conv1')
            net = tl.layers.Conv2d(net, filter_size*2, (3, 3), (2, 2),
                W_init=w_init, name='d_conv2')
            net = tl.layers.BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.3),
                is_train=is_train, gamma_init=gamma_init, name='d_bn1')
            net = tl.layers.Conv2d(net, filter_size*4, (3, 3), (2, 2),
                W_init=w_init, name='d_conv3')
            net = tl.layers.BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.3),
                is_train=is_train, gamma_init=gamma_init, name='d_bn2')
            net = tl.layers.Conv2d(net, filter_size*8, (3, 3), (2, 2),
                W_init=w_init, name='d_conv4')
            net = tl.layers.BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.3),
                is_train=is_train, gamma_init=gamma_init, name='d_bn3')
            net = tl.layers.FlattenLayer(net, name='d_flatten')
            net = tl.layers.DenseLayer(net, n_units=1, W_init=w_init, name='d_linear')

            logits = net.outputs
            net.outputs = tf.nn.sigmoid(net.outputs)

        return net, logits

    def _build_training(self):
        with tf.variable_scope('d_loss'):
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d2_logits), logits=self.d2_logits))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_logits), logits=self.d_logits))
            d_loss = d_loss_real + d_loss_fake

        with tf.variable_scope('g_loss'):
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_logits), logits=self.d_logits))

        opter = tf.train.AdamOptimizer(self.args.lr)

        def build_opt(loss, name):
            n_vars = tl.layers.get_variables_with_name(name, True, True)
            grads = tf.gradients(loss, n_vars)
            grads, _ = tf.clip_by_global_norm(grads, 0.01)
            opt = opter.apply_gradients(list(zip(grads, n_vars)))
            return opt

        g_opt = build_opt(g_loss, 'generator')
        d_opt = build_opt(d_loss, 'discriminator')

        return g_opt, d_opt, g_loss, d_loss

    def _build_tensorboard(self):
        self.err_d_tb = tf.placeholder(tf.float32, name='err_d_tb')
        self.err_g_tb = tf.placeholder(tf.float32, name='err_g_tb')

        tf.summary.scalar('errD', self.err_d_tb)
        tf.summary.scalar('errG', self.err_g_tb)

    def train(self, batch_images, batch_z):
        feed_dict = {self.r_input_ph: batch_images,
                     self.z_ph: batch_z}
        d_err, _ = self.sess.run([self.d_loss, self.d_opt], feed_dict)
    
        feed_dict = {self.z_ph: batch_z}
        for _ in range(self.args.g_update):
            g_err, _ = self.sess.run([self.g_loss, self.g_opt], feed_dict)

        feed_dict = {self.err_g_tb: g_err,
                     self.err_d_tb: d_err}

        self.time_step += 1
        summary = self.sess.run(self.merge_all, feed_dict)
        self.writer.add_summary(summary, self.time_step)

        return g_err, d_err

    def generate(self, z):
        imgs = self.sess.run(self.g2_net.outputs, feed_dict={self.z_ph: z})
        return imgs

class DataSet(object):
    def __init__(self, args, gan):
        self.args = args
        self.gan = gan

        self.data_files = glob(os.path.join("./data", args.dataset, "*.jpg"))
        self.sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(args.sample_size, args.z_dim)).astype(np.float32)

    def run(self):
        iter_counter = 0
        start_time = time.time()
        for e in range(self.args.epochs):
            shuffle(self.data_files)

            sample_files = self.data_files[0:self.args.sample_size]
            sample = [get_image(file, self.args.image_size, is_crop=self.args.is_crop, resize_w=self.args.image_size, is_grayscale=0) for file in sample_files]
            sample_images = np.array(sample).astype(np.float32)
            print("[*] Sample images updated! time per epoch: {}".format(time.time() - start_time))
            
            batch_idxs = min(len(self.data_files), self.args.train_size) // self.args.batch_size

            start_time = time.time()
            for idx in range(0, batch_idxs):
                batch_files = self.data_files[idx*self.args.batch_size:(idx+1)*self.args.batch_size]
                batch = [get_image(file, self.args.image_size, is_crop=self.args.is_crop, resize_w=self.args.image_size, is_grayscale=0) for file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_z = np.random.normal(loc=0.0, scale=1.0, size=(self.args.sample_size, self.args.z_dim)).astype(np.float32)

                g_err, d_err = self.gan.train(batch_images, batch_z)
                print("Epoch: [%2d/%2d] [%4d/%4d], d_loss: %.8f, g_loss: %.8f" \
                    % (e, self.args.epochs, idx, batch_idxs, d_err, g_err))

                iter_counter += 1
                if np.mod(iter_counter, self.args.sample_step) == 0:
                    img = self.gan.generate(self.sample_seed)
                    tl.visualize.save_images(img, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(self.args.sample_dir, e, idx))


if __name__ == '__main__':
    gan = DCGAN(args)
    dataset = DataSet(args, gan)

    dataset.run()


        
