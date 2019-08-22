"""Pull Problem.
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


def grasps_to_inputs(point_cloud, grasps):
    """Converts a list of grasps to an image and pose tensor.

    Args:
        point_cloud: The depth image of shape [num_frames, num_points, 3].
        grasps: Grasp vectors of shape [num_frames, num_grasps, 4].

    Returns:
        Dictionary of input numpy arrays.
    """

    return OrderedDict([
        ('point_cloud', point_cloud), ('grasps', grasps)])


class PullPointCloudProblem(problem.Problem):
    """Pull Problem."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 num_points=1024):
        """Initialize."""
        super(PullPointCloudProblem, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training)

        self.sess = None
        self.point_cloud_ph = tensor_spec.to_placeholder(
            self.time_step_spec.observation['point_cloud'])

        self.grasps_ph = tensor_spec.to_placeholder(
            self.action_spec['grasp'])

        self.tasks_ph = tensor_spec.to_placeholder(
            self.action_spec['task'])

        self.converted_data = grasps_to_inputs(
            tf.expand_dims(self.point_cloud_ph, 0),
            tf.expand_dims(self.grasps_ph, 0))

        spec_list = [
            tf.TensorSpec([num_points, 3],
                          tf.float32, 'point_cloud'),
            tf.TensorSpec([1, 4], tf.float32, 'grasps'),
            tf.TensorSpec([1], tf.int64, 'grasp_success'),
        ]
        self._spec = OrderedDict(
            [(spec.name, spec) for spec in spec_list])

    def convert_trajectory(self, trajectory):
        """Convert trajectory."""
        # Get the default session if it is not set.
        if self.sess is None:
            self.sess = tf.get_default_session()

        observation = trajectory.observation

        point_cloud = observation['point_cloud']
        action = trajectory.action

        grasps = action['grasp']
        grasp_4dof = grasps
        reward = trajectory.reward

        keypoints = action['keypoints'].flatten()
        keypoints = np.concatenate([[reward],
            keypoints], axis=0).reshape((1, -1))

        point_cloud = np.squeeze(point_cloud)
        x, y, z, angle = grasp_4dof
        grasp_4dof = np.array(
            [reward, x, y, z, 0, 0, angle],
            dtype=np.float32)

        extra_data = OrderedDict([
            ('point_cloud', point_cloud),
            ('grasp_4dof', grasp_4dof),
            ('keypoints', keypoints)])

        data = OrderedDict([
            ('point_cloud', point_cloud),
            ('grasps', grasps),
            ('grasp_success', trajectory.reward),
        ])
        return data, extra_data

    def preprocess(self, batch):
        """Proprocess the batch data."""
        return batch

    def loss(self, batch, outputs):
        """Get the loss function."""
        return tf.losses.sigmoid_cross_entropy(batch['grasp_success'],
                                               outputs['logit'])

    def add_summaries(self, batch, outputs):
        """Add summaries."""

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
