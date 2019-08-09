import tensorflow as tf
from cvae.sort import align
from cvae.network import Network


class GraspDiscriminator(Network):

    def build_model(self, x, g, keep_ratio=2,
                    aligned_point_cloud=False):
        with tf.variable_scope('grasp_discriminator',
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

            x = self.conv_layer(x, 16, name='conv1_1')
            x = self.conv_layer(x, 16, name='conv1_2')

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')

            x = self.conv_layer(x, 64, name='conv3_1')
            x = self.conv_layer(x, 64, name='conv3_2')

            x = self.conv_layer(x, 512, name='conv4_1')
            x = tf.reduce_max(x, [1, 2], keepdims=False)

            x = self.fc_layer(x, 256, name='fc1')
            x = self.fc_layer(x, 256, name='fc2')
            p = self.fc_layer(x, 1, linear=True, name='out')

        if aligned_point_cloud:
            return p, aligned
        else:
            return p


class KeypointDiscriminator(Network):

    def build_model(self, x, ks, group_size=512):
        """Builds the vae keypoint encoder

        Args:
            x: (B, N, 1, 3) Input point cloud
            ks: [(B, K1, 3), (B, K2, 3), ...]
            List of different type of keypoints

        Returns:
            p: (B, 1) Pre-sigmoid logit
        """

        with tf.variable_scope('keypoint_discriminator',
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

            ks_concat = tf.concat(ks, axis=1)
            x = self.group(tf.squeeze(x, 2), 
                           tf.squeeze(ks_concat, 2), 
                           group_size)
            x = x - ks_concat
            x = tf.transpose(x, [0, 2, 1, 3])

            x = self.conv_layer(x, 16, name='conv1_1')
            x = self.conv_layer(x, 16, name='conv1_2')

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')

            x = self.conv_layer(x, 32, name='conv5_1')
            x = self.conv_layer(x, 512, name='conv5_2')

            x = tf.reduce_max(x, axis=1, keepdims=False)

            xs = tf.split(x, num_keypoints, axis=1)
            xs = [tf.reduce_mean(x, axis=1) for x in xs]
            
            x = tf.concat(xs, axis=1)

            x = self.fc_layer(x, 256, name='fc1')
            x = self.fc_layer(x, 256, name='fc2')
            o = self.fc_layer(x, 1, linear=True, name='out')

        return o
