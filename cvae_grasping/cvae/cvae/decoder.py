import numpy as np
import tensorflow as tf
from cvae.sort import sort_tf

class Decoder(object):

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

    def down_sample(self, x, p, stride, thres, name):
        with tf.variable_scope(name):
            print('Original shape {}'.format(
                x.get_shape().as_list()))
            new_p = self.max_pool(p, [1, 1],
                    stride=[stride, 1], name=name)
            d = tf.expand_dims(
                    tf.transpose(new_p, 
                [0, 2, 3, 1]), 1) - tf.expand_dims(p, 4)
            d = tf.linalg.norm(d, axis=[2, 3], 
                    keepdims=False)
            _, h, w = d.get_shape().as_list()
            print('Down sample {}->{}'.format(h, w))

            d_mean = tf.reduce_mean(d, 
                    axis=1, keepdims=True)
            mask = tf.cast(tf.less(d, d_mean * thres), 
                    tf.float32)
            mask_sum = tf.reduce_sum(mask, 
                    axis=1, keepdims=True) + 1e-6
            x = tf.transpose(tf.squeeze(x, 2), [0, 2, 1])
            x = tf.matmul(x, mask) / mask_sum
            x = tf.expand_dims(
                    tf.transpose(x, [0, 2, 1]), 2)
            print('Down sampled shape {}'.format(
                x.get_shape().as_list()))
            return x, new_p


    def concat_xz(self, x, z):
        _, n, _, _ = x.get_shape().as_list()
        z_n = tf.tile(z, [1, n, 1, 1])
        return tf.concat([x, z_n], axis=3)
    
    def build_model(self, x, z):
        with tf.variable_scope('vae_decoder', 
                reuse=tf.AUTO_REUSE):

            miu, sigma = tf.split(z, 2, axis=1)
            z = miu + sigma * tf.random_normal(
                    tf.shape(sigma), 0.0, 1.0)
            _, z_dim = z.get_shape().as_list()
            z = tf.reshape(z, [-1, 1, 1, z_dim])
   
            mean_x = tf.reduce_mean(x, 
                    axis=1, keepdims=True)
            x = x - mean_x

            mean_r = tf.reduce_mean(
                         tf.linalg.norm(
                             x, axis=3, keepdims=True),
                         axis=1, keepdims=True)
            x = x / (mean_r + 1e-6)

            #x = tf.expand_dims(
            #        sort_tf(tf.squeeze(x, 2)), 2)
            p = x
        
            x = self.conv_layer(x, 16, name='conv1_1')
            x = self.conv_layer(x, 16, name='conv1_2')
            #x, p = self.down_sample(x, p, 4, 0.5, 'down_sample1')
            x = self.concat_xz(x, z)

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')
            #x, p = self.down_sample(x, p, 4, 0.5, 'down_sample2')
            x = self.concat_xz(x, z)

            x = self.conv_layer(x, 64, name='conv3_1')
            x = self.conv_layer(x, 256, name='conv3_2')
            x, p = self.down_sample(x, p, 64, 0.7, 'down_sample3')

            x = self.conv_layer(x, 256, name='conv5_1')

            location = self.conv_layer(x, 3,
                    name='conv5_2', linear=True) * (
                            mean_r + 1e-6)  + mean_x

            rotation = self.conv_layer(x, 6,
                    name='conv5_3', linear=True)

            x = tf.concat([location, rotation], axis=3)
            
            x = tf.squeeze(x, 2)
            g = tf.transpose(x, [0, 2, 1])
        
        return g
