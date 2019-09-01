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

    def build_model(self, x, ks, v=None):
        with tf.variable_scope('keypoint_discriminator',
                               reuse=tf.AUTO_REUSE):
            g_kp, f_kp = [tf.squeeze(k, 1) for k in ks]

            vx, vy, vz = tf.split(f_kp - g_kp, 3, axis=1)
            rz = tf.atan2(vy, vx)
            rx = tf.zeros_like(rz)
            pose_g = tf.concat([g_kp, rx, rx, rz], axis=1)
            pose_f = tf.concat([f_kp, rx, rx, rz], axis=1)

            pose_v = tf.concat([rx, rx, rx, rx, rx, rz], axis=1)

            x_g = tf.expand_dims(
                  align(tf.squeeze(x, 2), pose_g), 2)
            x_f = tf.expand_dims(
                  align(tf.squeeze(x, 2), pose_f), 2)

            x = tf.concat([x_g, x_f], axis=1)

            x = self.conv_layer(x, 16, name='conv1_1')
            x = self.conv_layer(x, 16, name='conv1_2')

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')

            x = self.conv_layer(x, 64, name='conv3_1')
            x = self.conv_layer(x, 64, name='conv3_2')

            x = self.conv_layer(x, 512, name='conv4_1')

            x_g, x_f = tf.split(x, 2, axis=1)

            x_g = tf.reduce_max(x_g, [1, 2], keepdims=False)
            x_f = tf.reduce_max(x_f, [1, 2], keepdims=False)

            x = tf.concat([x_g, x_f], axis=1)

            if v is not None:
                x_v = align(v, pose_v)
                x_v = self.fc_layer(x_v, 32, name='v_fc1')
                x_v = self.fc_layer(x_v, 64, name='v_fc2')
                x_v = self.fc_layer(x_v, 128, name='v_fc3')
                x = tf.concat([x, x_v], axis=1)

            x = self.fc_layer(x, 256, name='fc1')
            x = self.fc_layer(x, 256, name='fc2')
            p = self.fc_layer(x, 1, linear=True, name='out')

            return p


class ActionDiscriminator(Network):

    def build_model(self, x, ats):
        with tf.variable_scope('action_discriminator',
                               reuse=tf.AUTO_REUSE):
            grasp, trans, rot = ats

            x_g = x - tf.reshape(grasp, [-1, 1, 1, 3])

            vx, vy = tf.split(rot, 2, axis=1)
            rz = tf.atan2(vy, vx)
            zero = tf.zeros_like(rz)
            pose_t = tf.concat(
                    [zero, zero, zero, zero, zero, rz], axis=1)
            x_t = tf.expand_dims(
                  align(tf.squeeze(x_g, 2), pose_t), 2)
            x_t = x_t + tf.reshape(
                    tf.concat([trans, zero], axis=1), [-1, 1, 1, 3])

            x = tf.concat([x_g, x_t], axis=1)

            x = self.conv_layer(x, 16, name='conv1_1')
            x = self.conv_layer(x, 16, name='conv1_2')

            x = self.conv_layer(x, 32, name='conv2_1')
            x = self.conv_layer(x, 32, name='conv2_2')

            x = self.conv_layer(x, 64, name='conv3_1')
            x = self.conv_layer(x, 64, name='conv3_2')

            x = self.conv_layer(x, 512, name='conv4_1')

            x_g, x_t = tf.split(x, 2, axis=1)

            x_g = tf.reduce_max(x_g, [1, 2], keepdims=False)
            x_t = tf.reduce_max(x_t, [1, 2], keepdims=False)

            x = tf.concat([x_g, x_t], axis=1)

            x = self.fc_layer(x, 256, name='fc1')
            x = self.fc_layer(x, 256, name='fc2')
            p = self.fc_layer(x, 1, linear=True, name='out')

            return p

