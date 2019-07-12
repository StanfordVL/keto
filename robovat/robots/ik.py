"""Inverse kinematics (IK) solvers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from third_party import ikpy

from robovat.math import Pose


class IKSovler(object):
    """Base class for IK solvers.
    """

    def __init__(self, path, active_links=None):
        pass

    def inverse_kinematics(self, target, ee_link=-1, initial_positions=None):
        """Compute the IK results.
        """
        raise NotImplementedError


class ScipyIKSovler(IKSovler):
    """IK solver implemented with scipy.
    """

    def __init__(self, path, active_links=None):
        self._chain = ikpy.Chain.from_urdf_file(path, active_links=active_links)

    def inverse_kinematics(self, target, ee_link=-1, initial_positions=None):
        """Compute the IK results.
        """
        assert ee_link == -1 or ee_link == len(self._chain.links) - 1

        ik_result = self._chain.inverse_kinematics(
                target=target.matrix4, initial_positions=initial_positions)

        return ik_result.tolist()

    def forward_kinematics(self, positions):
        """Compute the forward kinematics.
        """
        matrix4 = self._chain.forward_kinematics(positions)
        return Pose(matrix4)
