"""Planning network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def center_point_cloud(point_cloud, name=None):
    """Convert point cloud into centers and centered point clouds.

    Args:
        point_cloud: Array of shape
            [batch_size, num_movables, num_points, 3].

    Returns:
        point: Array of [batch_size, num_movables, num_points, 3].
        center: Array of [batch_size, num_movables, 3].
    """
    with tf.variable_scope(name, default_name='convert_point_cloud'):
        center = tf.reduce_mean(point_cloud, axis=-2, name='center')
        point = point_cloud - tf.expand_dims(center, -2)
        point = tf.identity(point, name='point')
        return point, center
