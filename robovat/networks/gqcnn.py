"""Grasp Quality Convolutional Neural Network (GQCNN).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import tensorflow as tf
import gin.tf
from tf_agents.networks import network

import numpy as np  # NOQA
import matplotlib.pyplot as plt  # NOQA

from robovat.networks.resnet_utils import resnet_block_layer
from robovat.utils.logging import logger

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


# The default batch norm params.
BATCH_NORM_PARAMS = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': ['moving_vars'],
            'moving_variance': ['moving_vars'],
            },
        'center': True,
        'scale': True,
        }


# The default instance norm params.
INSTANCE_NORM_PARAMS = {
        'center': True,
        'scale': True,
        }


# The default layer norm params.
LAYER_NORM_PARAMS = {
        'center': True,
        'scale': True,
        }


def plot_batched_images(images):
    plt.figure(figsize=(10, 10))
    num_images = images.shape[0]
    d = int(np.ceil(np.sqrt(num_images)))

    for i in range(num_images):
        plt.subplot(d, d, i + 1)
        plt.imshow(images[i].squeeze(axis=-1), cmap=plt.cm.gray_r)
        plt.title('id: %d' % (i + 1))

    plt.tight_layout()
    plt.show()

    return images


def visualize(grasp_images, images):
    print_ops = [
        # tf.py_func(plot_batched_images, [images], tf.float32),
        tf.py_func(plot_batched_images, [grasp_images], tf.float32),
    ]
    print_op = tf.group(print_ops)
    with tf.control_dependencies([print_op]):
        return tf.identity(grasp_images)


def grasps_to_inputs(images, grasps, crop_size, crop_scale):
    """Converts a list of grasps to an image and pose tensor.

    Args:
        images: The depth image of shape [num_images, height, width, 1].
        grasps: Grasp vectors of shape [num_images, num_grasps, 5].
        crop_size: The list [height, width] of the image crop.
        crop_scale: The scaling factor of the image crop.

    Returns:
        Dictionary of input numpy arrays.
    """
    num_images = int(images.get_shape()[0])
    image_height = int(images.get_shape()[1])
    image_width = int(images.get_shape()[2])
    num_grasps = int(grasps.get_shape()[1])

    if crop_scale != 1.0:
        size = tf.cast(tf.shape(images)[1:3], tf.float32) * crop_scale
        size = tf.cast(size, tf.int32)
        images = tf.image.resize_bilinear(images, size, name='resized_images')
    else:
        size = tf.constant([image_height, image_width], dtype=tf.int32)

    # Grasps.
    grasps = tf.reshape(grasps, [num_images * num_grasps, 5], 'reshaped_grasps')
    grasp_centers = 0.5 * (grasps[:, 0:2] + grasps[:, 2:4])
    grasp_depths = grasps[:, 4:5]
    grasp_angles = tf.math.atan2(
        grasps[:, 3] - grasps[:, 1], grasps[:, 2] - grasps[:, 0])

    # Tile the images.
    images = tf.expand_dims(images, axis=1, name='expanded_images')
    images = tf.tile(images, [1, num_grasps, 1, 1, 1], 'tiled_images')
    images = tf.reshape(images,
                        [num_images * num_grasps, size[0], size[1], 1],
                        name='reshaped_image')

    # Image transformation.
    image_center = tf.constant([[0.5 * image_width, 0.5 * image_height]])
    translations = (image_center - grasp_centers) * crop_scale
    images = tf.contrib.image.translate(images, translations)
    images = tf.contrib.image.rotate(images, grasp_angles)

    # Take the crops.
    boxes = tf.stack([-0.5 * crop_size[0], -0.5 * crop_size[1],
                      0.5 * crop_size[0], 0.5 * crop_size[1]])
    scale = tf.stack([image_height, image_width, image_height, image_width])
    scale = tf.divide(1.0, tf.cast(scale, tf.float32))
    boxes = 0.5 + boxes * scale
    boxes = tf.tile(tf.expand_dims(boxes, 0), [num_images * num_grasps, 1])
    box_ind = tf.range(0, num_images * num_grasps)

    grasp_images = tf.image.crop_and_resize(
        images,
        boxes,
        box_ind,
        crop_size=crop_size,
        method='bilinear',
        name='grasp_images')

    # grasp_images = visualize(grasp_images, images)

    return OrderedDict([
        ('grasp_image', grasp_images), ('grasp_pose', grasp_depths)])


@gin.configurable
class GQCNN(network.Network):
    """Grasp Quality Convolutional Neural Network (GQCNN)."""
    def __init__(self,
                 time_step_spec,
                 action_spec,
                 crop_size=(32, 32),
                 crop_scale=1.0,
                 depth_scale=100.0,
                 name='GQCNN'):
        """Initialize."""
        self._crop_size = crop_size
        self._crop_scale = crop_scale
        self._depth_scale = depth_scale
        super(GQCNN, self).__init__(
            observation_spec=(),
            action_spec=action_spec,
            state_spec=(),
            name=name)

        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

    def call(self,
             observation,
             action,
             network_state=(),
             is_training=False):
        """Call the network in the policy."""
        images = observation['depth']
        grasps = action
        inputs = grasps_to_inputs(
            images, grasps, self._crop_size, self._crop_scale)
        outputs = self.forward(inputs, is_training)

        num_images = images.get_shape()[0]
        num_grasps = grasps.get_shape()[1]

        # TODO(kuanfang): Enable multiple images in CEM.
        assert num_images == 1

        return tf.reshape(outputs['grasp_success'], [num_images, num_grasps])

    def forward(self, inputs, is_training):
        """Forward the network in the policy or on the dataset."""
        # TODO(kuanfang): Make sure tf.AUTO_REUSE is the right way to go.
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            grasp_image = inputs['grasp_image']
            grasp_pose = inputs['grasp_pose']
            outputs = dict()

            with tf.variable_scope('inputs'):
                pixel_mean = tf.reduce_mean(grasp_image, axis=[1, 2, 3],
                                            keep_dims=True)
                grasp_image = (grasp_image - pixel_mean) * self._depth_scale
                pixel_mean = tf.squeeze(pixel_mean, axis=[1, 2])
                grasp_pose = (grasp_pose - pixel_mean) * self._depth_scale
                logger.debug('Input grasp_image has shape: %s',
                             grasp_image.get_shape())
                logger.debug('Input grasp_pose has shape: %s',
                             grasp_pose.get_shape())

            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                with slim.arg_scope(
                        [slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=(
                            tf.truncated_normal_initializer(0.0, 0.01)),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=BATCH_NORM_PARAMS):
                    with slim.arg_scope(
                            [slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=(
                                tf.truncated_normal_initializer(0.0, 0.01)),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=BATCH_NORM_PARAMS):

                        # Image stream.
                        with tf.variable_scope('image_conv'):

                            net = slim.conv2d(grasp_image, 64, [6, 6],
                                              scope='conv1')
                            net = slim.max_pool2d(net, [2, 2], stride=2,
                                                  scope='pool1')
                            net = resnet_block_layer(net, 64, stride=1,
                                                     shortcut_fn=None,
                                                     scope='res1')
                            net = resnet_block_layer(net, 128, stride=2,
                                                     shortcut_fn='projection',
                                                     scope='res2')
                            net = resnet_block_layer(net, 256, stride=2,
                                                     shortcut_fn='projection',
                                                     scope='res3')

                        # Final conv
                        net = slim.conv2d(net, 8, [1, 1], scope='final_conv')
                        net = slim.flatten(net, scope='final_conv_flatten')

                        # Grasp prediction.
                        with tf.variable_scope('grasp'):
                            # Image FC feature.
                            net = slim.fully_connected(net, 64,
                                                       scope='image_fc')
                            image_fc = net

                            # Pose stream.
                            net = slim.fully_connected(grasp_pose, 8,
                                                       scope='pose_fc')
                            pose_fc = net

                            # Merged stream.
                            net = tf.concat(
                                    [image_fc, pose_fc],
                                    axis=-1)
                            net = slim.fully_connected(net, 64, scope='fc1')

                            net = slim.fully_connected(net,
                                                       1,
                                                       activation_fn=None,
                                                       normalizer_fn=None,
                                                       normalizer_params=None,
                                                       scope='logit')
                            outputs['logit'] = net

                            # Outputs.
                            net = tf.sigmoid(net, name='grasp_success')
                            outputs['grasp_success'] = net

        return outputs
