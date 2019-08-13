import tensorflow as tf
from cvae.network import Network


class GraspDecoder(Network):

    def build_model(self, x, z):
        with tf.variable_scope('vae_grasp_decoder',
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

            p = x

            x = self.conv_layer(x, 16, name='conv1_1')
            x = self.conv_layer(x, 16, name='conv1_2')
            x = self.concat_xz(x, z)

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')
            x = self.concat_xz(x, z)

            x = self.conv_layer(x, 64, name='conv3_1')
            x = self.conv_layer(x, 256, name='conv3_2')
            x, p = self.down_sample(x, p, 64, 0.7, 'down_sample')

            x = self.conv_layer(x, 256, name='conv5_1')

            location = self.conv_layer(x, 3,
                                       name='conv5_2', linear=True) * (
                mean_r + 1e-6) + mean_x

            rotation = self.conv_layer(x, 6,
                                       name='conv5_3', linear=True)

            x = tf.concat([location, rotation], axis=3)

            x = tf.squeeze(x, 2)
            g = tf.transpose(x, [0, 2, 1])

        return g


class KeypointDecoder(Network):

    def build_model(self, x, z, nv=0, 
                    truncated_normal=False):
        with tf.variable_scope('vae_keypoint_decoder',
                               reuse=tf.AUTO_REUSE):

            miu, sigma = tf.split(z, 2, axis=1)
            if truncated_normal:
                z = miu + sigma * tf.random.truncated_normal(
                    tf.shape(sigma), 0.0, 1.0)
            else:
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

            p = x

            x = self.conv_layer(x, 16, name='conv1_1')
            x = self.conv_layer(x, 16, name='conv1_2')
            x = self.concat_xz(x, z)

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')
            x = self.concat_xz(x, z)

            x = self.conv_layer(x, 64, name='conv3_1')
            x = self.conv_layer(x, 256, name='conv3_2')
            x, p = self.down_sample(x, p, 64, 0.7, 'down_sample')

            x_f = self.conv_layer(x, 256, name='conv6_1')
            x_f = tf.reduce_max(x_f, axis=1, keepdims=True)
            functional_keypoints = self.conv_layer(x_f, 3,
                                                   name='conv6_2',
                                                   linear=True) * (
                mean_r + 1e-6) + mean_x

            x_g = self.conv_layer(x, 256, name='conv7_1')
            x_g = tf.reduce_max(x_g, axis=1, keepdims=True)
            grasping_keypoints = self.conv_layer(x_g, 3,
                                                 name='conv7_2',
                                                 linear=True) * (
                mean_r + 1e-6) + mean_x

            if nv:
                x_v = self.conv_layer(x, 256, name='conv8_1')
                x_v = tf.reduce_max(x_v, axis=1, keepdims=True)
                x_v = self.conv_layer(x_v, 3 * nv, name='conv8_2',
                                      linear=True)
                funct_vect = tf.reshape(x_v, [-1, nv, 3])
            else:
                funct_vect = None

            keypoints = [grasping_keypoints,
                         functional_keypoints]
            keypoints = [tf.squeeze(k, 2) for k in keypoints]
        return keypoints, funct_vect

