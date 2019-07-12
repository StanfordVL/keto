"""Parallel jaw grasps.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.math import get_transform
from robovat.math import Pose


class Grasp(object):
    """Parallel-jaw grasp in image space.

    """

    def __init__(self, pose, width=0.0):
        """Initialize.

        Args:
            pose: Pose of the grasp, with the center of the jaw tips as
                the position and the orientation of the gripper as the
                orientation.
            width: Distance between the jaws in meters.
        """
        self.pose = pose
        self.width = width

    @property
    def endpoints(self):
        """Returns the two endpoints of jaws."""
        offset = np.dot([float(self.width_pixel) / 2, 0., 0.],
                        np.self.center.matrix3.T)
        p1 = self.center.position + offset
        p2 = self.center.position - offset
        return p1, p2

    @property
    def width_pixel(self, camera_matrix):
        """Width in pixels.
        """
        raise NotImplementedError


class Grasp2D(object):
    """Parallel-jaw grasp in image space.

    """

    def __init__(self, center, angle, depth, width=0.0, camera=None):
        """Initialize.

        Args:
            center: Point (x, y) in image space.
            angle: Grasp axis angle with the camera x-axis.
            depth: Depth of the grasp center in 3D space.
            width: Distance between the jaws in meters.
            camera: The camera sensor for projection and deprojection.
        """
        self.center = center
        self.angle = angle
        self.depth = depth
        self.width = width
        self.camera = camera

    @property
    def axis(self):
        """Grasp axis.
        """
        return np.array([np.cos(self.angle), np.sin(self.angle)])

    @property
    def endpoints(self):
        """Grasp endpoints.
        """
        p1 = self.center - (float(self.width_pixel) / 2) * self.axis
        p2 = self.center + (float(self.width_pixel) / 2) * self.axis
        return p1, p2

    @property
    def width_pixel(self):
        """Width in pixels.
        """
        if self.camera is None:
            raise ValueError('Must specify camera intrinsics to compute gripper'
                             'width in 3D space.')

        # form the jaw locations in 3D space at the given depth
        p1 = np.array([0, 0, self.depth])
        p2 = np.array([self.width, 0, self.depth])

        # project into pixel space
        u1 = self.camera.project_point(p1, is_world_frame=False)
        u2 = self.camera.project_point(p2, is_world_frame=False)

        return np.linalg.norm(u1 - u2)

    @property
    def pose(self):
        """Computes the 3D pose of the grasp relative to the camera.

        If an approach direction is not specified then the camera
        optical axis is used.

        Returns:
            The pose of the grasp in the camera frame.
        """
        if self.camera is None:
            raise ValueError('Must specify camera intrinsics to compute 3D '
                             'grasp pose.')

        # Compute 3D grasp center in camera basis.
        grasp_center_camera = self.camera.deproject_pixel(
                self.center, self.depth, is_world_frame=False)

        # Compute 3D grasp axis in camera basis.
        grasp_axis_image = self.axis
        grasp_axis_image = grasp_axis_image / np.linalg.norm(grasp_axis_image)
        grasp_axis_camera = np.array(
            [grasp_axis_image[0], grasp_axis_image[1], 0])
        grasp_axis_camera = (
            grasp_axis_camera / np.linalg.norm(grasp_axis_camera))

        # Aligned with camera Z axis.
        grasp_x_camera = np.array([0, 0, 1])
        grasp_y_camera = grasp_axis_camera
        grasp_z_camera = np.cross(grasp_x_camera, grasp_y_camera)
        grasp_x_camera = np.cross(grasp_z_camera, grasp_y_camera)
        grasp_rot_camera = np.array(
            [grasp_x_camera, grasp_y_camera, grasp_z_camera]).T

        if np.linalg.det(grasp_rot_camera) < 0:
            # Fix possible reflections due to SVD.
            grasp_rot_camera[:, 0] = -grasp_rot_camera[:, 0]

        pose = Pose([grasp_center_camera, grasp_rot_camera])

        return pose

    @property
    def vector(self):
        """Returns the feature vector for the grasp.

        v = [p1, p2, depth], where p1 and p2 are the jaw locations in image
        space.
        """
        p1, p2 = self.endpoints
        return np.r_[p1, p2, self.depth]

    def as_4dof(self):
        """Computes the 4-DOF pose of the grasp in the world frame.

        Returns:
            The 4-DOF gripper pose in the world.
        """
        angle = (self.angle + np.pi / 2)
        grasp_pose_in_camera = Pose([self.pose.position, [0, 0, angle]])
        grasp_pose_in_world = get_transform(source=self.camera.pose).transform(
            grasp_pose_in_camera)

        x, y, z = grasp_pose_in_world.position
        angle = grasp_pose_in_world.euler[2]

        return [x, y, z, angle]

    @staticmethod
    def from_vector(value, width=0.0, camera=None):
        """Creates a Grasp2D instance from a feature and additional parameters.

        Args:
            value: Feature vector.
            width: Grasp opening width, in meters.
            camera: The camera sensor for projection and deprojection.
        """
        # Read feature vector.
        p1 = value[:2]
        p2 = value[2:4]
        depth = value[4]

        # Compute center and angle.
        center = (p1 + p2) / 2
        axis = p2 - p1
        angle = np.arctan2(axis[1], axis[0])

        return Grasp2D(center, angle, depth, width, camera)

    @staticmethod
    def image_dist(g1, g2, alpha=1.0):
        """Computes the distance between grasps in image space.

        Euclidean distance with alpha weighting of angles

        Args:
            g1: First grasp.
            g2: Second grasp.
            alpha: Weight of angle distance (rad to meters).

        Returns:
            Distance between grasps.
        """
        # Point to point distances.
        point_dist = np.linalg.norm(g1.center - g2.center)

        # Axis distances.
        axis_dist = np.arccos(g1.axis.dot(g2.axis))

        return point_dist + alpha * axis_dist
