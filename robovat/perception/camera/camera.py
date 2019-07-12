"""Base class for Cameras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from robovat.math import get_transform
from robovat.math import Pose
from robovat.math import Orientation


class Camera(object):
    """Abstract base class for cameras.
    """

    def start(self):
        """Starts the camera stream.
        """
        pass

    def stop(self):
        """Stops the camera stream.

        Returns:
            True if succeed, False if fail.
        """
        return True

    def reset(self):
        """Restarts the camera stream.
        """
        self.stop()
        self.start()

    def frames(self):
        """Get the latest set of frames.

        Returns:
            A dictionary of RGB image, depth image and segmentation image.
            'image': The RGB image as an uint8 np array of [width, height, 3].
            'depth': The depth image as a float32 np array of [width, height].
            'segmask': None value.
        """
        raise NotImplementedError

    def load_calibration(self, path, robot_pose=[[0, 0, 0], [0, 0, 0]]):
        """Set the camera by using the camera calibration results.

        Args:
            path: The data directory of the calibration results.
        """
        intrinsics_path = os.path.join(path, 'IR_intrinsics.npy')
        intrinsics = np.load(intrinsics_path, encoding='latin1')
        translation_path = os.path.join(path, 'robot_IR_translation.npy')
        translation = np.load(translation_path, encoding='latin1')
        rotation_path = os.path.join(path, 'robot_IR_rotation.npy')
        rotation = np.load(rotation_path, encoding='latin1')

        # Convert the extrinsics from the robot frame to the world frame.
        from_robot_to_world = get_transform(source=robot_pose)
        robot_pose_in_camera = Pose([translation, rotation])
        camera_pose_in_robot = robot_pose_in_camera.inverse()
        camera_pose_in_world = from_robot_to_world.transform(
                camera_pose_in_robot)
        world_origin_in_camera = camera_pose_in_world.inverse()
        translation = world_origin_in_camera.position
        rotation = world_origin_in_camera.matrix3

        return intrinsics, translation, rotation

    def set_calibration(self, intrinsics, translation, rotation):
        """Set the camera calibration data.

        Args:
            intrinsics: The intrinsics matrix.
            translation: The translation vector.
            rotation: The rotation matrix.
        """
        self._intrinsics = np.array(intrinsics).reshape((3, 3))
        self._translation = np.array(translation).reshape((3,))
        self._rotation = Orientation(rotation).matrix3

    def project_point(self, point, is_world_frame=True):
        """Projects a point cloud onto the camera image plane.

        Args:
            point: 3D point to project onto the camera image plane.
            is_world_frame: True if the 3D point is defined in the world frame,
                False if it is defined in the camera frame.

        Returns:
            pixel: 2D pixel location in the camera image.
        """
        point = np.array(point)

        if is_world_frame:
            point = np.dot(point - self.pose.position, self.pose.matrix3)

        projected = np.dot(point, self.intrinsics.T)
        projected = np.divide(projected, np.tile(projected[2], [3]))
        projected = np.round(projected)
        pixel = np.array(projected[:2]).astype(np.int16)

        return pixel

    def deproject_pixel(self, pixel, depth, is_world_frame=True):
        """Deprojects a single pixel with a given depth into a 3D point.

        Args:
            pixel: 2D point representing the pixel location in the camera image.
            depth: Depth value at the given pixel location.
            is_world_frame: True if the 3D point is defined in the world frame,
                False if it is defined in the camera frame.

        Returns:
            point: The deprojected 3D point.
        """
        point = depth * np.linalg.inv(self.intrinsics).dot(np.r_[pixel, 1.0])

        if is_world_frame:
            point = self.pose.position + np.dot(point, self.pose.matrix3.T)

        return point

    @property
    def pose(self):
        world_origin_in_camera = Pose([self._translation, self._rotation])
        return world_origin_in_camera.inverse()

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def translation(self):
        return self._translation

    @property
    def rotation(self):
        return self._rotation

    @property
    def cx(self):
        return self.intrinsics[0, 2]

    @property
    def cy(self):
        return self.intrinsics[1, 2]
