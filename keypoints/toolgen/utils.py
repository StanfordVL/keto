import os
import tensorflow as tf
import matplotlib.pyplot as plt


def rotation_mat(rz):
    """Computes the rotation matrix wrt. the z axis.

    Args:
        rz: The rotation wrt. the z axis.

    Returns:
        The rotation matrix with shape (3, 3).
    """
    zero = tf.constant(0.0, dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    rot_mat = tf.concat([[tf.cos(rz)], [-tf.sin(rz)], [zero],
                         [tf.sin(rz)], [tf.cos(rz)], [zero],
                         [zero], [zero], [one]], axis=0)
    rot_mat = tf.reshape(rot_mat, [3, 3])
    return rot_mat


def visualize_keypoints(point_cloud,
                        keypoints,
                        prefix,
                        name,
                        plot_lim=2,
                        point_size=40):
    """Plots the point cloud and keypoints.
    
    Args:
        point_cloud: The point cloud with shape (N, 3).
        keypoints: A list of [grasp point, function point, effect point].
        prefix: The directory to save the figure.
        name: The name of the figure.
        plot_lim: The coordinate range of the plots.
        point_size: The size of each point in the point cloud.

    Returns:
        None.
    """
    [grasp_point, funct_point, funct_vect] = keypoints
    fig = plt.figure(figsize=(6, 6))
    xs = point_cloud[:, 0]
    ys = point_cloud[:, 1]

    ax = fig.add_subplot(111)
    ax.scatter(xs, ys, s=point_size, c='#4d4f53', marker='.')
    ax.set_axis_off()
    ax.grid(False)
    plt.axis('equal')
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.savefig(os.path.join(prefix, '{}_point_cloud.png'.format(name)))
    plt.close()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys, s=point_size, c='#4d4f53', marker='.')
    ax.set_axis_off()
    ax.grid(False)
    
    ax.scatter(grasp_point[:, 0],
               grasp_point[:, 1],
               s=point_size * 20,
               c='#eaab00')
    
    ax.scatter(funct_point[:, 0],
               funct_point[:, 1],
               s=point_size * 20,
               c='#8c1515')
    
    ax.scatter(funct_point[:, 0] + funct_vect[:, 0],
               funct_point[:, 1] + funct_vect[:, 1],
               s=point_size * 20,
               c='#007c92')
    plt.axis('equal')
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.savefig(os.path.join(prefix, '{}_keypoints.png'.format(name)))
    plt.close()
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.grid(False)
    
    ax.scatter(grasp_point[:, 0],
               grasp_point[:, 1],
               s=point_size * 15,
               c='#eaab00')
    
    ax.scatter(funct_point[:, 0],
               funct_point[:, 1],
               s=point_size * 15,
               c='#8c1515')
    
    ax.scatter(funct_point[:, 0] + funct_vect[:, 0],
               funct_point[:, 1] + funct_vect[:, 1],
               s=point_size * 15,
               c='#007c92')
    plt.axis('equal')
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.savefig(os.path.join(prefix, '{}_raw_keypoints.png'.format(name)))
    plt.close()
    return


def min_dist(x, y):
    """Computes the Chamfer distance between two point sets.
    
    Args:
        x: The first point set with shape (N, 3).
        y: The second point set with shape (M, 3).

    Return:
        min_dist: The Chamfer distance.
    """
    x = tf.reshape(x, [-1, 1, 3])
    y = tf.reshape(y, [1, -1, 3])
    dist = tf.reduce_sum(tf.square(x - y), axis=2)
    min_dist = tf.reduce_min(dist)
    return min_dist
