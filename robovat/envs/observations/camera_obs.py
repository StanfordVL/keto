"""Camera observation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from robovat.envs.observations import observation
from robovat.perception import point_cloud_utils


INF = 2**32 - 1
COLOR_MAP = ['r', 'y', 'b', 'g', 'm', 'k'] 


class CameraObs(observation.Observation):
    """Camera observation."""

    def __init__(self,
                 camera,
                 modality,
                 max_visible_distance_m=None,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_obs'
        self.camera = camera
        self.modality = modality
        self.max_visible_distance_m = max_visible_distance_m or INF

        self.env = None

    def initialize(self, env):
        self.env = env
        self.camera.start()

    def on_episode_start(self):
        """Called at the start of each episode."""
        self.camera.reset()

    def get_gym_space(self):
        """Returns gym space of this observation."""
        height = self.camera.height
        width = self.camera.width

        if self.modality == 'rgb':
            shape = (height, width, 3)
            return gym.spaces.Box(0, 255, shape, dtype=np.uint8)
        elif self.modality == 'depth':
            shape = (height, width, 1)
            return gym.spaces.Box(0.0, self.max_visible_distance_m, shape,
                                  dtype=np.float32)
        elif self.modality == 'segmask':
            shape = (height, width, 1)
            return gym.spaces.Box(0, int(INF), shape, dtype=np.uint32)
        else:
            raise ValueError('Unrecognized modality: %r.' % (self.modality))

    def get_observation(self):
        """Returns the observation data of the current step."""
        images = self.camera.frames()
        image = images[self.modality]

        if self.modality == 'rgb':
            pass
        elif self.modality == 'depth' or self.modality == 'segmask':
            image = image[:, :, np.newaxis]
        else:
            raise ValueError('Unrecognized modality: %r.' % (self.modality))

        # TODO(kuanfang): For debugging.
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()

        return image

class DeprojectParamsObs(CameraObs):
    """Point cloud observation."""

    def __init__(self,
                 camera,
                 num_points,
                 segment_table=True,
                 max_visible_distance_m=None,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_obs'
        self.camera = camera
        self.num_points = num_points
        self.segment_table = segment_table
        self.max_visible_distance_m = max_visible_distance_m or INF

        self.env = None

    def get_gym_space(self):
        """Returns gym space of this observation."""
        shape = (21, )
        return gym.spaces.Box(-INF, INF, shape, dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return self.camera.deproject_params


class PointCloudObs(CameraObs):
    """Point cloud observation."""

    def __init__(self,
                 camera,
                 num_points,
                 segment_table=True,
                 max_visible_distance_m=None,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_obs'
        self.camera = camera
        self.num_points = num_points
        self.segment_table = segment_table
        self.max_visible_distance_m = max_visible_distance_m or INF

        self.env = None

    def get_gym_space(self):
        """Returns gym space of this observation."""
        shape = (self.num_points, 3)
        return gym.spaces.Box(-INF, INF, shape, dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        images = self.camera.frames()
        point_cloud = self.camera.deproject_depth_image(images['depth'])

        if self.segment_table:
            point_cloud = point_cloud_utils.segment_table(point_cloud)

        point_cloud = point_cloud_utils.downsample(
                point_cloud, num_samples=self.num_points)

        # point_cloud_utils.plot_point_cloud(point_cloud, axis_limit=None)

        return point_cloud


class SegmentedPointCloudObs(CameraObs):
    """Point cloud observation."""

    def __init__(self,
                 camera,
                 num_points,
                 body_names,
                 max_visible_distance_m=None,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_obs'
        self.camera = camera
        self.num_points = num_points
        self.body_names = body_names
        self.max_visible_distance_m = max_visible_distance_m or INF

        self.env = None

    def on_episode_start(self):
        """Called at the start of each episode."""
        self.camera.reset()
        self.body_ids = [self.env.simulator.bodies[name].uid
                         for name in self.body_names]

    def get_gym_space(self):
        """Returns gym space of this observation."""
        num_bodies = len(self.body_names)
        shape = (num_bodies, self.num_points, 3)

        return gym.spaces.Box(-INF, INF, shape, dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        images = self.camera.frames()
        depth = images['depth']
        segmask = images['segmask']
        segmask = segmask.flatten()
        point_cloud = self.camera.deproject_depth_image(depth)
        point_cloud = point_cloud_utils.segment(
            point_cloud, segmask, self.body_ids, self.num_points)

        # point_cloud_utils.plot_point_cloud(
        #     point_cloud, c=COLOR_MAP, axis_limit=None)

        return point_cloud


class CameraIntrinsicsObs(observation.Observation):
    """Camera intrinsics observation.

    Returns the camera instrincs matrix.
    """

    def __init__(self,
                 camera,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_intrinsics_obs'
        self.camera = camera

        self.env = None

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Box(
            low=-INF * np.ones((3, 3)),
            high=INF * np.ones((3, 3)),
            dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return self.camera.intrinsics


class CameraTranslationObs(observation.Observation):
    """Camera translation observation.

    Returns the translation matrix of the camera extrinsics.
    """

    def __init__(self,
                 camera,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_translation_obs'
        self.camera = camera

        self.env = None

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Box(
            low=-INF * np.ones((3,)),
            high=INF * np.ones((3,)),
            dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return self.camera.translation


class CameraRotationObs(observation.Observation):
    """Camera translation observation.

    Returns the translation matrix of the camera extrinsics.
    """

    def __init__(self,
                 camera,
                 name=None):
        """Initialize."""
        self.name = name or 'camera_rotation_obs'
        self.camera = camera

        self.env = None

    def get_gym_space(self):
        """Returns gym space of this observation."""
        # TODO(kuanfang): Correct the space.
        return gym.spaces.Box(
            low=-INF * np.ones((3, 3)),
            high=INF * np.ones((3, 3)),
            dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return self.camera.rotation
