"""A toy environment of Brownian movement.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
import matplotlib.pyplot as plt

from robovat.envs import robot_env
from robovat.envs.observations import observation
from robovat.utils.logging import logger  # NOQA


INF = 2e24


class MovableObs(observation.Observation):
    """Observation."""

    def __init__(self, name=None):
        """Initialize."""
        self.name = name or 'movable'
        self.env = None

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Box(-INF, INF, [self.env.config.NUM_MOVABLES, 2],
                              dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return self.env.movable


class BrownianEnv(robot_env.RobotEnv):
    """The environment of robot arm."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=False):
        """Initialize."""
        self.movable = None
        observations = [MovableObs()]
        reward_fns = []
        self.is_action_valid = None

        super(BrownianEnv, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=simulator,
            config=config,
            debug=debug)

        low = [self.config.BOUNDARY.LEFT, self.config.BOUNDARY.UP]
        high = [self.config.BOUNDARY.RIGHT, self.config.BOUNDARY.DOWN]
        self.action_space = gym.spaces.Box(
            low=np.array(low),
            high=np.array(high),
            dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            'movable_id': gym.spaces.Discrete(self.config.NUM_MOVABLES),
            'motion': gym.spaces.Box(
                low=-np.array([0.5, 0.5]),
                high=np.array([0.5, 0.5]),
                dtype=np.float32)
        })

    def reset_scene(self):
        """Reset the scene in simulation or the real world."""
        low = [self.config.BOUNDARY.LEFT, self.config.BOUNDARY.UP]
        low = np.array(low, dtype=np.float32)
        low = np.tile(np.expand_dims(low, 0), [self.config.NUM_MOVABLES, 1])
        high = [self.config.BOUNDARY.RIGHT, self.config.BOUNDARY.DOWN]
        high = np.array(high, dtype=np.float32)
        high = np.tile(np.expand_dims(high, 0), [self.config.NUM_MOVABLES, 1])
        self.movable = np.random.uniform(low=low, high=high)

    def execute_action(self, action):
        """Execute the robot action.
        """
        movable = self.movable
        movable_id = action['movable_id']
        motion = action['motion']

        prev_movable = self.movable.copy()
        self.movable[movable_id, :] += motion
        self.movable += np.random.uniform(-0.01, 0.01, [3, 2])

        if self.debug:
            plt.figure(figsize=(5, 5))
            plt.xlim([self.config.BOUNDARY.LEFT, self.config.BOUNDARY.RIGHT])
            plt.ylim([self.config.BOUNDARY.UP, self.config.BOUNDARY.DOWN])

            # Plot movables.
            for i in range(movable.shape[0]):
                if i == 0:
                    color = 'r'
                elif i == 1:
                    color = 'g'
                elif i == 2:
                    color = 'b'
                else:
                    color = 'k'

                plt.scatter(prev_movable[i, 0], prev_movable[i, 1],
                            c=color, s=100, marker='+')
                plt.scatter(self.movable[i, 0], self.movable[i, 1],
                            c=color, s=10)
                plt.plot([prev_movable[i, 0], self.movable[i, 0]],
                         [prev_movable[i, 1], self.movable[i, 1]],
                         c=color)

            plt.show()
