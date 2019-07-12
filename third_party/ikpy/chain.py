# coding= utf8
"""
.. module:: chain
This module implements the Chain class.
"""

import numpy as np

from . import urdf_utils
from . import inverse_kinematics


class Chain(object):
    """The base Chain class

    :param list links: List of the links of the chain
    :param list active_links: A list of boolean indicating that whether or
        not the corresponding link is active.
    """
    def __init__(self, links, active_links=None):
        self.links = links
        self._length = sum([link._length for link in links])
        # Avoid length of zero in a link
        for (index, link) in enumerate(self.links):
            if link._length == 0:
                link._axis_length = self.links[index - 1]._axis_length

        # If the active_links is not given, set it to True for every link.
        if active_links is not None:
            if len(active_links) != len(self.links):
                raise ValueError('Active links mask length of %d != '
                                 'number of links %d'
                                 % (len(active_links), len(self.links)))
            self.active_links = np.array(active_links)
            # Always set the last link to False.
            self.active_links[-1] = False
        else:
            self.active_links = np.array([True] * len(links))

    def active_to_full(self, active_joints, initial_positions):
        full_joints = np.array(initial_positions, copy=True, dtype=np.float)
        np.place(full_joints, self.active_links, active_joints)
        return full_joints

    def active_from_full(self, joints):
        return np.compress(self.active_links, joints, axis=0)

    def forward_kinematics(self, positions):
        """Returns the transformation matrix of the forward kinematics

        Note : Inactive positions must be in the list.

        :param list positions: The list of the positions of each joint.
        :returns: The transformation matrix
        """
        frame_matrix = np.eye(4)

        if len(self.links) != len(positions):
            raise ValueError('positions vector length %d != number of links %d'
                             % (len(positions), len(self.links)))

        for index, (link, joint_angle) in enumerate(zip(self.links, positions)):
            # Compute iteratively the position
            # NB : Use asarray to avoid old sympy problems
            frame_matrix = np.dot(
                    frame_matrix, 
                    np.asarray(link.get_transformation_matrix(joint_angle)))

        return frame_matrix

    def inverse_kinematics(self, target, initial_positions=None, **kwargs):
        """Computes the inverse kinematic on the specified target

        :param numpy.array target: The frame target of the inverse kinematic, in
            meters. It must be 4x4 transformation matrix.
        :param numpy.array initial_positions: Optional : the initial position of
            each joint of the chain. Defaults to 0 for each joint.
        :returns: The list of the positions of each joint according to the
            target. Note : Inactive joints are in the list.
        """
        # Checks on input
        target = np.array(target)
        if target.shape != (4, 4):
            raise ValueError('Your target must be a 4x4 transformation matrix')

        if initial_positions is None:
            initial_positions = [0] * len(self.links)

        return inverse_kinematics.inverse_kinematic_optimization(
                self, target, initial_positions=initial_positions, **kwargs)

    @classmethod
    def from_urdf_file(cls, urdf_file, base_elements=['base_link'],
                       last_link_offset=None, active_links=None):
        """Creates a chain from an URDF file.

        :param urdf_file: The path of the URDF file
        :type urdf_file: string
        :param base_elements: List of the links beginning the chain
        :type base_elements: list of strings
        :param last_link_offset: Optional : The translation vector of the tip.
        :type last_link_offset: numpy.array
        :param list active_links: The active links
        """
        links = urdf_utils.get_urdf_parameters(
                urdf_file, base_elements=base_elements,
                last_link_offset=last_link_offset)

        # Add an origin link at the beginning
        return cls(links, active_links=active_links)
