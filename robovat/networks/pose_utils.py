"""Planning network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def batched_matrix_dot_vector(matrix, vector):
    return tf.squeeze(tf.matmul(matrix, tf.expand_dims(vector, -1)), -1)


def transform(data, rotation, translation):
    """Rigid transformation of points.

    Args:
        data: Float array of shape [..., 3].
        rotation: Float array of shape [..., 3, 3].
        translation: Float array of shape [..., 3].

    Returns:
        Float array of shape [..., 3].
    """
    assert data.get_shape()[-1] == 3, (
        'data is of shape %r' % (data.get_shape()))
    assert rotation.get_shape()[-2] == 3, (
        'rotation is of shape %r' % (rotation.get_shape()))
    assert rotation.get_shape()[-1] == 3, (
        'rotation is of shape %r' % (rotation.get_shape()))
    assert translation.get_shape()[-1] == 3, (
        'translation is of shape %r' % (translation.get_shape()))
    assert len(data.get_shape()) == len(rotation.get_shape()) - 1
    assert len(data.get_shape()) == len(translation.get_shape())

    return batched_matrix_dot_vector(rotation, data) + translation


def pose_to_rigid_transform(pose, inverse):
    """Compute a transformation matrix of the pose frame.

    Args: 
        pose: Float array of shape [..., 6] representing
            (x, y, z, roll, pitch, yaw).
        inverse: If True, compute the inverse transformation.

    Returns: 
        matrix4: Float array of shape [..., 3, 4].
    """
    assert pose.get_shape()[-1] == 6

    cx = tf.cos(pose[..., 3])
    cy = tf.cos(pose[..., 4])
    cz = tf.cos(pose[..., 5])

    sx = tf.sin(pose[..., 3])
    sy = tf.sin(pose[..., 4])
    sz = tf.sin(pose[..., 5])

    # [[cy * cz, sx * sy * cz - cx * sz, cx * sy * sz - sx * cz],
    #  [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
    #  [-sy, sx * cy, cx * cy]]
    rotation = tf.stack([
        tf.stack([cy * cz, cy * sz, -sy], axis=-1),
        tf.stack([sx * sy * cz - cx * sz, sx * sy * sz + cx * cz, sx * cy],
                 axis=-1),
        tf.stack([cx * sy * sz - sx * cz, cx * sy * sz - sx * cz, cx * cy],
                 axis=-1)
        ],
        axis=-1)
    translation = pose[..., 0:3]

    if inverse:
        rotation_T = tf.linalg.transpose(rotation)
        return rotation_T, batched_matrix_dot_vector(rotation_T, -translation)
    else:
        return rotation, translation

