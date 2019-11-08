import numpy as np
import tensorflow as tf


def sort_tf(x, is_training=tf.constant(False, dtype=tf.bool)):
    """ Sorts the point cloud wrt. the priciple dimension.
    
    Args:
        x: (B, N, 3) The point cloud.
    
    Returns:
        y: (B, N, 3) The sorted point cloud.
    """
    x_mean = tf.reduce_mean(
        x, axis=1, keep_dims=True)
    x = x - x_mean
    s, u, v = tf.svd(x)
    v = tf.expand_dims(v[:, :, 0], axis=1)
    v = tf.cond(is_training,
                lambda: random_noise(v), lambda: v)
    x_proj = tf.reduce_sum(x * v, axis=2)
    diff_mat = tf.add(
        tf.expand_dims(x_proj, axis=2),
        -tf.expand_dims(x_proj, axis=1))
    diff_mat = tf.cast(tf.greater(diff_mat, 0.0),
                       tf.float32)

    rank = tf.cast(tf.reduce_sum(diff_mat, axis=2),
                   tf.int32)
    perm_mat = tf.one_hot(rank,
                          depth=rank.get_shape().as_list()[1])

    y = tf.matmul(perm_mat, x + x_mean,
                  transpose_a=True)
    return y


def std(v):
    """Computes the standard deviation.

    Args:
        v: (B, 1, 3) The input data.

    Returns:
        o: (1, 1, 3) The std wrt. the first axis.
    """
    v = v - tf.reduce_mean(v, axis=0, keep_dims=True)
    o = tf.reduce_mean(v * v, axis=0, keep_dims=True)
    o = tf.sqrt(o)
    return o


def random_noise(v, std_ratio=0.5):
    """Adds random noise to unit vectors.

    Args:
        v: (B, 1, 3) The input unit vectors.

    Returns:
        vn: (B, 1, 3) The unit vectors with noise.
    """
    r = tf.random_normal(
        shape=tf.shape(v)) * std(v) * std_ratio
    v = v + r
    n = tf.norm(v, axis=2, keep_dims=True) + 1e-12
    vn = v / n
    return vn


def rotation_matrix(alpha, beta, gamma):
    """Computes the rotation matrix.

    Args:
        alpha: The rotation around x axis.
        beta: The rotation around y axis.
        gamma: The rotation around z axis.

    Returns:
        The rotation matrix.
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    R = np.dot(np.dot(Rz, Ry), Rx)
    return R


def rot_mat(rx, ry, rz):
    """Computes a batch of rotation matrices.

    Args:
        rx: (B, 1) The rotation around x axis.
        ry: (B, 1) The rotation around y axis.
        rz: (B, 1) The rotation around z axis.

    Returns:
        R: (B, 3, 3) The rotation matrix.
    """
    zeros = tf.zeros_like(rx)
    ones = tf.ones_like(rx)
    Rx = tf.concat([ones, zeros, zeros,
                    zeros, tf.cos(rx), -tf.sin(rx),
                    zeros, tf.sin(rx), tf.cos(rx)],
                   axis=1)
    Rx = tf.reshape(Rx, [-1, 3, 3])
    Ry = tf.concat([tf.cos(ry), zeros, tf.sin(ry),
                    zeros, ones, zeros,
                    -tf.sin(ry), zeros, tf.cos(ry)],
                   axis=1)
    Ry = tf.reshape(Ry, [-1, 3, 3])
    Rz = tf.concat([tf.cos(rz), -tf.sin(rz), zeros,
                    tf.sin(rz), tf.cos(rz), zeros,
                    zeros, zeros, ones],
                   axis=1)
    Rz = tf.reshape(Rz, [-1, 3, 3])
    R = tf.matmul(Rx, tf.matmul(Ry, Rz))
    return R


def align(x, p):
    """ Aligns the point cloud wrt. 6-DoF grasp.
    
    Args:
        x: (B, N, 3) The input point cloud.
        p: (B, 6) The 6-DoF grasps.
    
    Returns:
        y: (B, N, 3) The aligned point cloud.
    """
    c, rx, ry, rz = tf.split(p, [3, 1, 1, 1], axis=1)
    R = rot_mat(-rx, -ry, -rz)

    c = tf.expand_dims(c, 1)
    x = tf.transpose(x - c, [0, 2, 1])
    x = tf.matmul(R, x)
    y = tf.transpose(x, [0, 2, 1])
    return y
