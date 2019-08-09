"""DexNet Problem.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import matplotlib.pyplot as plt  # NOQA
import numpy as np  # NOQA
import tensorflow as tf
from tf_agents.specs import tensor_spec

from robovat.problems import problem
from robovat.perception import tf_depth_utils
from robovat.perception.camera import Camera
from robovat.grasp import Grasp2D

slim = tf.contrib.slim


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
    num_grasps = int(grasps.get_shape()[0])

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


class Grasp4DofProblem(problem.Problem):
    """DexNet Problem."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 crop_size=(32, 32),
                 crop_scale=1.0):
        """Initialize."""
        super(Grasp4DofProblem, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training)

        self.crop_size = crop_size
        self.crop_scale = crop_scale

        self.sess = None
        self.images_ph = tensor_spec.to_placeholder(
            self.time_step_spec.observation['depth'])
        self.grasps_ph = tensor_spec.to_placeholder(
            self.action_spec)
        self.converted_data = grasps_to_inputs(
            tf.expand_dims(self.images_ph, 0),
            tf.expand_dims(self.grasps_ph, 0),
            self.crop_size,
            self.crop_scale)

        spec_list = [
            tf.TensorSpec([self.crop_size[0], self.crop_size[1], 1],
                          tf.float32, 'grasp_image'),
            tf.TensorSpec([1], tf.float32, 'grasp_pose'),
            tf.TensorSpec([1], tf.int64, 'grasp_success'),
        ]
        self._spec = OrderedDict(
            [(spec.name, spec) for spec in spec_list])

    def decode_params(self, params):
        intrinsics, pose_position, pose_matrix = \
                np.split(np.squeeze(params),
                        [9, 12])
        intrinsics = np.reshape(intrinsics, [3, 3])
        pose_position = np.reshape(pose_position, [3, 1])
        pose_matrix = np.reshape(pose_matrix, [3, 3])
        return [intrinsics, pose_position, pose_matrix]

    def convert_trajectory(self, trajectory):
        """Convert trajectory."""
        # Get the default session if it is not set.
        if self.sess is None:
            self.sess = tf.get_default_session()

        observation = trajectory.observation
        images = observation['depth']
        point_cloud = observation['point_cloud']
        intrinsics = observation['intrinsics']
        translation = observation['translation']
        rotation = observation['rotation']

        camera = Camera(intrinsics=intrinsics,
                        translation=translation,
                        rotation=rotation)

        grasps = trajectory.action
        grasp2d = Grasp2D.from_vector(grasps, camera=camera)
        grasp_4dof = grasp2d.as_4dof()        
        reward = trajectory.reward

        point_cloud = np.squeeze(point_cloud)
        x, y, z, angle = grasp_4dof
        grasp_4dof = np.array(
                [reward, x, y, z, 0, 0, angle], 
                dtype=np.float32)
        extra_data = OrderedDict([
            ('point_cloud', point_cloud),
            ('grasp_4dof', grasp_4dof)])

        feed_dict = {
            self.images_ph: images,
            self.grasps_ph: grasps,
        }
        inputs = self.sess.run(self.converted_data, feed_dict=feed_dict)
        data =  OrderedDict([
            ('grasp_image', inputs['grasp_image']),
            ('grasp_pose', inputs['grasp_pose']),
            ('grasp_success', trajectory.reward)])
        return data, extra_data

    def preprocess(self, batch):
        """Proprocess the batch data."""
        if self.is_training:
            grasp_image = batch['grasp_image']
            grasp_image = tf_depth_utils.gamma_noise(grasp_image)
            grasp_image = tf_depth_utils.gaussian_noise(grasp_image, prob=0.5)
            grasp_image = tf_depth_utils.random_flip(grasp_image, prob=0.5)
            batch['grasp_image'] = grasp_image
        return batch

    def loss(self, batch, outputs):
        """Get the loss function."""
        return tf.losses.sigmoid_cross_entropy(batch['grasp_success'],
                                               outputs['logit'])

    def add_summaries(self, batch, outputs):
        """Add summaries."""
        tf.summary.image('grasp_image', batch['grasp_image'], max_outputs=8)

        labels = batch['grasp_success']
        predicted_scores = outputs['grasp_success']
        predicted_labels = tf.cast(tf.greater(predicted_scores, 0.5), tf.int64)

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
                'grasp/AUC': slim.metrics.streaming_auc(
                    predicted_scores, labels),
                'grasp/accuracy': slim.metrics.streaming_accuracy(
                    predicted_labels, labels),
                'grasp/precision': slim.metrics.streaming_precision(
                    predicted_labels, labels),
                'grasp/recall': slim.metrics.streaming_recall(
                    predicted_labels, labels),
        })

        for metric_name, metric_value in names_to_values.items():
            tf.summary.scalar(metric_name, metric_value)

        tf.summary.histogram('targets/grasp_success', labels)
        tf.summary.histogram('outputs/grasp_success', predicted_labels)

        for name, value in outputs.items():
            tf.summary.histogram(name, value)

        eval_ops = names_to_updates.values()

        return eval_ops
