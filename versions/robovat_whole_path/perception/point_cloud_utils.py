"""Utilities for point cloud data.

Copied from https://github.com/kuanfang/pointnet-lite.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # NOQA

from robovat.utils.logging import logger

try:
    import pcl  # NOQA
except Exception:
    print('Cannot import pcl.')

try:
    import pandas as pd
except Exception:
    print('Cannot import pandas.')

try:
    from pyntcloud import PyntCloud
except Exception:
    print('Cannot import pyntcloud.')


def downsample(point_cloud, num_samples):
    """Downsample a point cloud.

    Args:
        point_cloud: 3D point cloud of shape [num_points, 3].
        num_samples: Number of points to keep.

    Returns:
        downsampled_point_cloud: Downsampled 3D point cloud of shape
            [num_samples, 3].
    """
    num_points = point_cloud.shape[0]
    replace = num_points < num_samples
    inds = np.random.choice(np.arange(num_points), size=num_samples,
                            replace=replace)
    downsampled_point_cloud = point_cloud[inds]
    return downsampled_point_cloud


def segment(point_cloud, segmask, body_ids, num_samples):
    """Downsample a point cloud.

    Args:
        point_cloud: 3D point cloud of shape [num_points, 3].
        segmask: Integer array of shape [num_points].
        body_ids: List of segmentation IDs of the target bodies.
        num_samples: Number of points to keep for each body.

    Returns:
        segmented_point_cloud: Segmented 3D point cloud of shape
            [num_bodies, num_samples, 3].
    """
    num_bodies = len(body_ids)
    segmented_point_cloud = np.zeros([num_bodies, num_samples, 3],
                                     dtype=np.float32)

    for i in range(num_bodies):
        body_id = body_ids[i]
        inds_i = np.where(segmask == body_id)[0]
        if len(inds_i) > 0:
            point_cloud_i = point_cloud[inds_i]
            point_cloud_i = downsample(point_cloud_i, num_samples)
            segmented_point_cloud[i] = point_cloud_i
        else:
            logger.warning('No points were found for object % d.' % i)

    return segmented_point_cloud


def segment_table(point_cloud, thresh=0.01):
    """Remove all table points using ransac.

    Args:
        point_cloud: (num_points x 3) 3D point cloud.
        thresh: Maximum distance to allow in ransac.

    Returns:
        segmented_cloud: 3D point cloud without table points of shape
            [num_segmented_points, 3].
    """
    # PCL version.
    # num_points = point_cloud.shape[0]
    # cloud = pcl.PointCloud(point_cloud.astype(np.float32))
    # segmenter = cloud.make_segmenter()
    # segmenter.set_model_type(pcl.SACMODEL_PLAnum_pointsE)
    # segmenter.set_method_type(pcl.SAC_RAnum_pointsSAC)
    # segmenter.set_distance_threshold(thresh)
    # indices, model = segmenter.segment()
    # obj_idxs = np.setdiff1d(np.arange(num_points), indices)
    # segmented_cloud = point_cloud[obj_idxs]

    # PyntCloud version.
    cloud = PyntCloud(pd.DataFrame({'x': point_cloud[:, 0],
                                    'y': point_cloud[:, 1],
                                    'z': point_cloud[:, 2]}))
    cloud.add_scalar_field("plane_fit", max_dist=thresh)
    point_cloud = cloud.points.values
    segmented_cloud = point_cloud[point_cloud[:, -1] == 0][:, :3]

    return segmented_cloud


def plot_point_cloud(point_cloud,
                     c='b',
                     axis_limit=None):
    """Plot point cloud and return subplot handle.

    Usage:
        import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        plt.title('Point cloud')
        ax = fig.add_subplot(111, projection='3d')
        plot_point_cloud(ax, point_cloud)
        plt.show()
    """
    fig = plt.figure()
    plt.title('Point Cloud')
    ax = fig.add_subplot(111, projection='3d')

    if axis_limit is not None:
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_zlim(-axis_limit, axis_limit)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    if len(point_cloud.shape) == 2:
        xs = point_cloud[:, 0]
        ys = point_cloud[:, 1]
        zs = point_cloud[:, 2]
        ax.scatter(xs, ys, zs, s=1, c=c)
    elif len(point_cloud.shape) == 3:
        for i in range(point_cloud.shape[0]):
            xs = point_cloud[i, :, 0]
            ys = point_cloud[i, :, 1]
            zs = point_cloud[i, :, 2]
            color = c[i]
            ax.scatter(xs, ys, zs, s=1, c=color)
    else:
        raise ValueError

    plt.show()
