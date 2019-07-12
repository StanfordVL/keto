"""Observation of object pose.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from robovat.envs.observations import observation


INF = 2**32 - 1


class PoseObs(observation.Observation):
    """Pose observation."""

    def __init__(self,
                 body_names,
                 modality='pose',
                 name=None):
        """Initialize."""
        self.name = name or 'pose_obs'
        self.body_names = body_names
        self.modality = modality
        self.env = None

    def initialize(self, env):
        self.env = env
        assert self.env.simulator is not None

    def on_episode_start(self):
        """Called at the start of each episode."""
        if isinstance(self.body_names, list):
            self.bodies = [self.env.simulator.bodies[name]
                           for name in self.body_names]
        else:
            self.bodies = self.env.simulator.bodies[self.body_names]

    def get_gym_space(self):
        """Returns gym space of this observation."""
        if isinstance(self.body_names, list):
            _shape = (len(self.body_names),)
        else:
            _shape = ()

        if self.modality == 'pose':
            return gym.spaces.Box(-INF, INF, _shape + (6,), dtype=np.float32)
        elif self.modality == 'position':
            return gym.spaces.Box(-INF, INF, _shape + (3,), dtype=np.float32)
        elif self.modality == 'yaw_cossin':
            return gym.spaces.Box(-INF, INF, _shape + (2,), dtype=np.float32)
        elif self.modality == 'pose2d':
            return gym.spaces.Box(-INF, INF, _shape + (3,), dtype=np.float32)
        else:
            raise ValueError('Unrecognized modality: %r.' % (self.modality))

    def get_observation(self):
        """Returns the observation data of the current step."""
        if isinstance(self.body_names, list):
            return np.stack([self.get_pose(body) for body in self.bodies])
        else:
            return self.get_pose(self.bodies)

    def get_pose(self, body):
        """Returns the observation data of the current step."""
        pose = body.pose

        if self.modality == 'pose':
            return body.pose.to_array()
        elif self.modality == 'position':
            return pose.position
        elif self.modality == 'yaw_cossin':
            return np.array([np.cos(pose.yaw), np.sin(pose.yaw)],
                            dtype=np.float32)
        if self.modality == 'pose2d':
            return np.array([pose.x, pose.y, pose.yaw], dtype=np.float32)
        else:
            raise ValueError('Unrecognized modality: %r.' % (self.modality))
