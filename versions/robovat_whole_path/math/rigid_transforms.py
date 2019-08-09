"""Rigid transformation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.math.pose import Pose


def get_transform(source=None, target=None):
    """
    Get rigid transformation from one frame to another.

    Args:
        source: The source frame. Set to None if it is the world frame.
        target: The target frame. Set to None if it is the world frame.

    Returns:
        An instance of Pose.
    """
    if source is not None and not isinstance(source, Pose):
        source = Pose(source)

    if target is not None and not isinstance(target, Pose):
        target = Pose(target)

    if source is not None and target is not None:
        orientation = np.dot(target.matrix3.T, source.matrix3)
        position = np.dot(source.position - target.position, target.matrix3)
    elif source is not None:
        orientation = source.matrix3
        position = source.position
    elif target is not None:
        orientation = target.matrix3.T
        position = np.dot(-target.position, target.matrix3)
    else:
        orientation = np.eye(3, dtype=np.float32)
        position = np.ones(3, dtype=np.float32)

    return Pose([position, orientation])
