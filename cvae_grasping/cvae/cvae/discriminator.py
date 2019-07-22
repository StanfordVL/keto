import numpy as np
import tensorflow as tf
from cvae.sort import align

class Discriminator(object):

    def __init__(self):
        return

    def batch_norm_layer(self, x, eps=0.01):
        dimension = x.get_shape().as_list()[-1]
        mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
        beta = tf.get_variable('bta',
                               dimension,
                               tf.float32,
                               initializer= \
               tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gma',
                                dimension,
                                tf.float32,
                                initializer= \
                tf.constant_initializer(1.0, tf.float32))
        x = tf.nn.batch_normalization(x, mean,
                                variance, beta,
                                gamma, eps)
        return x

    def conv_layer(self, x, out_channels, 
                   kernel_size=1, dilation=1, 
                   linear=False, name=None):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(
                x, out_channels, 
                kernel_size=kernel_size, 
                dilation_rate=dilation,
                padding='same')
            if linear:
                return x
            x = tf.nn.relu(x)
            x = self.batch_norm_layer(x)
            return x

    def fc_layer(self, x, out_size, 
                 linear=False, name=None):
        with tf.variable_scope(name):
            x = tf.contrib.layers.flatten(x)
            x = tf.contrib.layers.fully_connected(
                 x, out_size, activation_fn=None)
            if linear:
                return x
            x = tf.nn.relu(x)
            return x

    def max_pool(self, x, name):
        with tf.variable_scope(name):
            x = tf.contrib.layers.max_pool2d(
                x, kernel_size=2, padding='SAME')
            return x
    
    def build_model(self, x, g, keep_ratio=2, 
                    aligned_point_cloud=False):
        with tf.variable_scope('discriminator',
                 reuse=tf.AUTO_REUSE):
            x = tf.expand_dims(
                    align(tf.squeeze(x, 2), g), 2)
            _, n, _, _ = x.get_shape().as_list()

            mean_r = tf.reduce_mean(
                         tf.linalg.norm(
                             x, axis=3, keepdims=True),
                         axis=1, keepdims=True)
            x = x / (mean_r + 1e-6)
    
            aligned = tf.squeeze(x, 2)

            #distance = tf.linalg.norm(aligned, axis=2)
            #_, indices = tf.math.top_k(-distance, 
            #        k=n//keep_ratio)
            #print('Indices shape {}'.format(
            #    indices.get_shape().as_list()))
            #one_hot = tf.one_hot(
            #            indices, depth=n,
            #            on_value=1.0, off_value=0.0,
            #            dtype=tf.float32)
            #mask = tf.reduce_sum(one_hot, axis=1)
            #mask = tf.reshape(mask, [-1, n, 1, 1])

            x = self.conv_layer(x, 16, name='conv1_1')
            x = self.conv_layer(x, 16, name='conv1_2')

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')

            x = self.conv_layer(x, 64, name='conv3_1')
            x = self.conv_layer(x, 64, name='conv3_2')

            x = self.conv_layer(x, 512, name='conv4_1')

            #x = x * mask

            x = tf.reduce_max(x, [1, 2], keepdims=False)

            x = self.fc_layer(x, 256, name='fc1')
            x = self.fc_layer(x, 256, name='fc2')
            p = self.fc_layer(x, 1, linear=True, name='out')

        if aligned_point_cloud:
            return p, aligned
        else:
            return p
