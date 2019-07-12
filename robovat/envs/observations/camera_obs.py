"""Camera observation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from robovat.envs.observations import observation


INF = 2**32 - 1


class CameraObs(observation.Observation):
    """Camera observation."""
    
    def __init__(self,
                 name,
                 camera,
                 modality,
                 max_visible_distance_m=None):
        """Initialize."""
        self.name = name 
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
            shape = (height, width)
            return gym.spaces.Box(0.0, self.max_visible_distance_m, shape,
                                  dtype=np.float32)
        elif self.modality == 'segmask':
            shape = (height, width)
            return gym.spaces.Box(0, int(INF), shape, dtype=np.uint32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        images = self.camera.frames()
        image = images[self.modality]

        # TODO(kuanfang): For debugging.
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()

        return image


class CameraIntrinsicsObs(observation.Observation):
    """Camera intrinsics observation.

    Returns the camera instrincs matrix.
    """
    
    def __init__(self,
                 name,
                 camera):
        """Initialize."""
        self.name = name 
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
