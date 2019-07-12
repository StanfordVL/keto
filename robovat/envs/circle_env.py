"""A toy environment of circles.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
import matplotlib.pyplot as plt

from robovat.envs import robot_env
from robovat.envs.observations import observation
from robovat.envs.reward_fns import reward_fn
from robovat.utils.logging import logger


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


class ValidObs(observation.Observation):
    """Observation."""

    def __init__(self, name=None):
        """Initialize."""
        self.name = name or 'is_valid'
        self.env = None

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Discrete(1)

    def get_observation(self):
        """Returns the observation data of the current step."""
        attrib = self.env.is_action_valid
        if attrib is True:
            value = 1
        elif attrib is False:
            value = 0
        else:
            raise ValueError('Unrecognized flag value: %r' % (attrib))
        return np.array(value, dtype=np.int64)


class ValidReward(reward_fn.RewardFn):
    """Reward function."""
    
    def __init__(self,
                 streaming_length=10000,
                 name=None):
        """Initialize."""
        self.name = name or 'valid_reward'
        self.streaming_length = streaming_length
        self.env = None
        self.history = []

    def get_reward(self):
        """Returns the reward value of the current step."""
        success = self.env.is_action_valid

        self._update_history(success)
        success_rate = np.mean(self.history or [-1])
        logger.debug('Success: %r, Success Rate %.3f',
                     success, success_rate)

        return success, True

    def _update_history(self, success):
        self.history.append(success)
        if len(self.history) > self.streaming_length:
            self.history = self.history[-self.streaming_length:]


class CircleEnv(robot_env.RobotEnv):
    """The environment of robot arm."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=False):
        """Initialize."""
        self.movable = None
        observations = [MovableObs()]
        # observations = [MovableObs(),
        #                 ValidObs()]
        reward_fns = [ValidReward()]
        self.is_action_valid = None

        super(CircleEnv, self).__init__(
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
            'position': gym.spaces.Box(
                low=np.array(low),
                high=np.array(high),
                dtype=np.float32),
        })

    def reset_scene(self):
        """Reset the scene in simulation or the real world."""
        low = [self.config.BOUNDARY.LEFT, self.config.BOUNDARY.UP]
        low = np.array(low, dtype=np.float32)
        low = np.tile(np.expand_dims(low, 0), [self.config.NUM_MOVABLES, 1])
        high = [self.config.BOUNDARY.RIGHT, self.config.BOUNDARY.DOWN]
        high = np.array(high, dtype=np.float32)
        high = np.tile(np.expand_dims(high, 0), [self.config.NUM_MOVABLES, 1])
        
        if self.config.USE_FIXED_MOVABLES:
            self.movable = np.array([
                [-0.5, 0.5], [0, -0.5], [0.5, 0.5],
                # [-0.3, 0.3]
            ], dtype=np.float32)
        else:
            self.movable = np.random.uniform(low=low, high=high)

        self.is_action_valid = 0

    def execute_action(self, action):
        """Execute the robot action.
        """
        movable = self.movable
        position = action['position']

        if position.ndim == 1:
            positions = [position]
        elif position.ndim == 2:
            positions = position
        else:
            raise ValueError('position.shape: %r, position.ndim: %r'
                             % (position.shape, position.ndim))

        valid_flags = []

        for i in range(len(positions)):
            position = positions[i]
            dists = np.linalg.norm(movable - position, axis=-1)
            is_reachable = dists <= self.config.REACHABLE_DIST
            is_reachable = np.any(is_reachable)
            is_collided = dists <= self.config.COLLISION_DIST
            is_collided = np.any(is_collided)
            is_position_valid = is_reachable and (not is_collided)
            valid_flags.append(is_position_valid)

        self.is_action_valid = valid_flags[0]

        if self.debug:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.gca()
            plt.xlim([self.config.BOUNDARY.LEFT, self.config.BOUNDARY.RIGHT])
            plt.ylim([self.config.BOUNDARY.UP, self.config.BOUNDARY.DOWN])

            # Plot movables.
            for i in range(movable.shape[0]):
                pos = movable[i]
                circle1 = plt.Circle((pos[0], pos[1]),
                                     self.config.REACHABLE_DIST,
                                     color='g', alpha=.2, fill=True)
                circle2 = plt.Circle((pos[0], pos[1]),
                                     self.config.COLLISION_DIST,
                                     color='r', alpha=.2, fill=True)
                ax.add_artist(circle1)
                ax.add_artist(circle2)

            # Plot positions.
            for i in range(len(positions)):
                position_color = 'g' if valid_flags[i] else 'r'
                position = positions[i]
                plt.scatter(position[0], position[1],
                            c=position_color, marker='+', s=100)

            plt.show()


class FixedCircleEnv(CircleEnv):
    """The environment of robot arm."""
