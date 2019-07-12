# coding= utf8
"""
.. module:: link
This module implements the Link class.
"""

import numpy as np
import sympy

from . import geometry_utils


class Link(object):
    """Base Link class.

    :param name: The name of the link
    :type name: string
    :param bounds: Optional : The bounds of the link. Defaults to None
    :type bounds: tuple
    :param use_symbolic_matrix: wether the transformation matrix is stored as
        Numpy array or as a Sympy symbolic matrix.
    :type use_symbolic_matrix: bool
    """

    def __init__(self, name, bounds=(None, None)):
        self.bounds = bounds
        self.name = name

    def _get_rotation_axis(self):
        # Defaults to None
        return [0, 0, 0, 1]

    def get_transformation_matrix(self, theta):
        raise NotImplementedError


class OriginLink(Link):
    """The link at the origin of the robot
    """
    def __init__(self):
        Link.__init__(self, name='Base link')
        self._length = 1

    def _get_rotation_axis(self):
        return [0, 0, 0, 1]

    def get_transformation_matrix(self, theta):
        return np.eye(4)


class URDFLink(Link):
    """Link in URDF representation.

    :param name: The name of the link
    :type name: string
    :param bounds: Optional : The bounds of the link. Defaults to None
    :type bounds: tuple
    :param translation: The translation vector. (In URDF, attribute "xyz"
        of the "origin" element).
    :type translation: numpy.array
    :param orientation: The orientation of the link. (In URDF, attribute "rpy"
        of the "origin" element).
    :type orientation: numpy.array
    :param rotation: The rotation axis of the link. (In URDF, attribute "xyz" of
        the "axis" element).
    :type rotation: numpy.array
    :param angle_representation: Optionnal : The representation used by the
    angle. Currently supported representations : rpy. Defaults to rpy, the URDF
        standard.
    :type angle_representation: string
    :param use_symbolic_matrix: wether the transformation matrix is stored as a
        Numpy array or as a Sympy symbolic matrix.
    :type use_symbolic_matrix: bool
    :returns: The link object
    :rtype: URDFLink
    :Example:

    URDFlink()
    """

    def __init__(self,
                 name,
                 translation,
                 orientation,
                 rotation,
                 bounds=(None, None),
                 angle_representation='rpy',
                 use_symbolic_matrix=True):

        Link.__init__(self, name=name, bounds=bounds)
        self.use_symbolic_matrix = use_symbolic_matrix
        self.translation = np.array(translation)
        self.orientation = np.array(orientation)
        self.rotation = np.array(rotation)

        self._length = np.linalg.norm(translation)
        self._axis_length = self._length

        if use_symbolic_matrix:
            # Angle symbolique qui paramètre la rotation du joint en cours
            theta = sympy.symbols('theta')

            symbolic_frame_matrix = np.eye(4)

            # Apply translation matrix
            symbolic_frame_matrix = symbolic_frame_matrix * sympy.Matrix(
                    geometry_utils.homogeneous_translation_matrix(*translation))

            # Apply orientation matrix
            symbolic_frame_matrix = (
                    symbolic_frame_matrix * 
                    geometry_utils.cartesian_to_homogeneous(
                        geometry_utils.rpy_matrix(*orientation))
                    )

            # Apply rotation matrix
            symbolic_frame_matrix = (
                symbolic_frame_matrix *
                geometry_utils.cartesian_to_homogeneous(
                    geometry_utils.symbolic_axis_rotation_matrix(
                        rotation, theta), matrix_type='sympy')
                )

            self.symbolic_transformation_matrix = sympy.lambdify(
                    theta, symbolic_frame_matrix, 'numpy')

    def _get_rotation_axis(self):
        return np.dot(
                geometry_utils.homogeneous_translation_matrix(
                        *self.translation),
                np.dot(geometry_utils.cartesian_to_homogeneous(
                           geometry_utils.rpy_matrix(*self.orientation)),
                       geometry_utils.cartesian_to_homogeneous_vectors(
                           self.rotation * self._axis_length))
                )

    def get_transformation_matrix(self, theta):
        if self.use_symbolic_matrix:
            frame_matrix = self.symbolic_transformation_matrix(theta)
        else:
            # Init the transformation matrix
            frame_matrix = np.eye(4)

            # First, apply translation matrix
            frame_matrix = np.dot(
                    frame_matrix,
                    geometry_utils.homogeneous_translation_matrix(
                        *self.translation)
                    )

            # Apply orientation
            frame_matrix = np.dot(
                    frame_matrix,
                    geometry_utils.cartesian_to_homogeneous(
                        geometry_utils.rpy_matrix(*self.orientation))
                    )

            # Apply rotation matrix
            frame_matrix = np.dot(
                    frame_matrix,
                    geometry_utils.cartesian_to_homogeneous(
                        geometry_utils.axis_rotation_matrix(
                            self.rotation, theta))
                    )

        return frame_matrix
