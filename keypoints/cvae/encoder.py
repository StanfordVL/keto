import tensorflow as tf
from cvae.network import Network


class GraspEncoder(Network):

    def build_model(self, x, g):

        with tf.variable_scope('vae_grasp_encoder',
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
            g_loc = (g_loc - mean_x) / (mean_r + 1e-6)
            g = tf.concat([g_loc, g_rot], axis=3)

            _, n, _, _ = x.get_shape().as_list()
            g = tf.tile(g, [1, n, 1, 1])

            x = self.conv_layer(x, 16, name='conv1_1')
            x = self.conv_layer(x, 16, name='conv1_2')

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')
            x = tf.concat([x, g], axis=3)

            x = self.conv_layer(x, 32, name='conv5_1')
            x = self.conv_layer(x, 512, name='conv5_2')

            x = tf.reduce_max(x, axis=[1, 2], keepdims=False)

            x = self.fc_layer(x, 256, name='fc1')
            x = self.fc_layer(x, 256, name='fc2')
            z = self.fc_layer(x, 4, linear=True, name='out')

            miu, sigma = tf.split(z, 2, axis=1)
            sigma_sp = tf.nn.softplus(sigma)
            z = tf.concat([miu, sigma_sp], axis=1)
        return z


class KeypointEncoder(Network):

    def build_model(self, x, ks):
        """Builds the vae keypoint encoder

        Args:
            x: (B, N, 1, 3) Input point cloud
            ks: [(B, K1, 3), (B, K2, 3), ...]
            List of different type of keypoints

        Returns:
            z: (B, 2*D) Mean and var of the latent
            variable with dimension D

        """

        with tf.variable_scope('vae_keypoint_encoder',
                               reuse=tf.AUTO_REUSE):

            mean_x = tf.reduce_mean(x,
                                    axis=1, keepdims=True)
            x = x - mean_x

            mean_r = tf.reduce_mean(
                tf.linalg.norm(
                    x, axis=3, keepdims=True),
                axis=1, keepdims=True)
            x = x / (mean_r + 1e-6)

            ks = [tf.expand_dims(k, axis=2) for k in ks]

            num_keypoints = [k.get_shape().as_list()[1] for k in ks]

            ks = [(k - mean_x) / (mean_r + 1e-6) for k in ks]

            _, num_points, _, _ = x.get_shape().as_list()

            x = tf.concat([x] + ks, axis=1)

            x = self.conv_layer(x, 16, name='conv1_1')
            x = self.conv_layer(x, 16, name='conv1_2')

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')

            x = self.conv_layer(x, 32, name='conv5_1')
            x = self.conv_layer(x, 256, name='conv5_2')

            x_ks = tf.split(x, [num_points] + num_keypoints, axis=1)
            x, ks = x_ks[0], x_ks[1:]

            x = tf.reduce_max(x, axis=[1, 2], keepdims=False)

            ks = [tf.reduce_max(k, axis=[1, 2], keepdims=False) for k in ks]
            x = tf.concat([x] + ks, axis=1)

            x = self.fc_layer(x, 256, name='fc1')
            x = self.fc_layer(x, 256, name='fc2')
            z = self.fc_layer(x, 4, linear=True, name='out')

            miu, sigma = tf.split(z, 2, axis=1)
            sigma_sp = tf.nn.softplus(sigma)
            z = tf.concat([miu, sigma_sp], axis=1)
        return z
