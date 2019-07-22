import numpy as np
import tensorflow as tf

class Encoder(object):

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

    def max_pool(self, x, ksize, stride, name):
        with tf.variable_scope(name):
            x = tf.contrib.layers.max_pool2d(
                x, kernel_size=ksize, 
                stride=stride, padding='same')
            return x
    
    def build_model(self, x, g):

        with tf.variable_scope('vae_encoder',
                reuse=tf.AUTO_REUSE):

            mean_x = tf.reduce_mean(x, 
                         axis=1, keepdims=True)
            x = x - mean_x

            mean_r = tf.reduce_mean(
                         tf.linalg.norm(
                             x, axis=3, keepdims=True),
                         axis=1, keepdims=True)
            x = x / (mean_r + 1e-6)

            g = tf.reshape(g, [-1, 1, 1, 9])
            g_loc, g_rot = tf.split(g, [3, 6], axis=3)
            g_loc = g_loc - mean_x
            g = tf.concat([g_loc, g_rot], axis=3)

            _, n, _, _ = x.get_shape().as_list()
            g = tf.tile(g, [1, n, 1, 1])

            x = self.conv_layer(x, 16, name='conv1_1')
            x = self.conv_layer(x, 16, name='conv1_2')

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')
            x = tf.concat([x, g], axis=3)

            x = self.conv_layer(x, 32,  name='conv5_1')
            x = self.conv_layer(x, 512, name='conv5_2')

            x = tf.reduce_max(x, axis=[1, 2], keepdims=False)
        
            x = self.fc_layer(x, 256, name='fc1')
            x = self.fc_layer(x, 256, name='fc2')
            z = self.fc_layer(x, 4, linear=True, name='out')

            miu, sigma = tf.split(z, 2, axis=1)
            sigma_sp = tf.nn.softplus(sigma)
            z = tf.concat([miu, sigma_sp], axis=1)
        return z

