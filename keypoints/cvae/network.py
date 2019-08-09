import tensorflow as tf


class Network(object):

    def __init__(self):
        return

    def batch_norm_layer(self, x, eps=0.01):
        dimension = x.get_shape().as_list()[-1]
        mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
        beta = tf.get_variable(
            'bta',
            dimension,
            tf.float32,
            initializer=tf.constant_initializer(
                0.0,
                tf.float32))
        gamma = tf.get_variable(
            'gma',
            dimension,
            tf.float32,
            initializer=tf.constant_initializer(
                1.0,
                tf.float32))
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
                kernel_size=[kernel_size, 1],
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

    def group(self, x, p, size):
        """ Groups the point cloud
            Args:
                x: (B, N, 3) Input point cloud
                p: (B, K, 3) Grouping centers
                size: () Number of points in a group
            Return:
                y: (B, K, size, 3) Grouped point cloud
        """
        _, n, _ = x.get_shape().as_list()
        _, k, _ = p.get_shape().as_list()
        dist = tf.linalg.norm(
                tf.expand_dims(x, axis=3) - 
                tf.transpose(tf.expand_dims(p, axis=3),
                    [0, 3, 2, 1]), axis=2)
        dist = tf.transpose(dist, [0, 2, 1])
        _, top_indices = tf.math.top_k(-dist, k=size)
        one_hot = tf.one_hot(top_indices, depth=n)
        mask = tf.greater(tf.reduce_sum(one_hot, axis=2), 0)
        y = tf.tile(tf.expand_dims(x, 1), [1, k, 1, 1])
        y = tf.boolean_mask(y, mask)
        y = tf.reshape(y, [-1, k, size, 3])
        return y

    def shape(self, x):
        print(x.get_shape().as_list())
        return
