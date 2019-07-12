"""The environment of flying hammer-shape tool.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
from matplotlib import pyplot as plt

from robovat.envs import robot_env
from robovat.envs.observations import observation
from robovat.utils.logging import logger


INF = 2**32 - 1


class StateObs(observation.Observation):
    """Pose observation."""

    def __init__(self,
                 name=None):
        """Initialize."""
        self.name = name or 'state_obs'
        self.env = None

    def initialize(self, env):
        self.env = env

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Box(-INF, INF, (3,), dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return np.array(self.env.robot_pose, dtype=np.float32)


class PolylineObs(observation.Observation):
    """Pose observation."""

    def __init__(self,
                 name=None):
        """Initialize."""
        self.name = name or 'polyline_obs'
        self.env = None

    def initialize(self, env):
        self.env = env

    def get_gym_space(self):
        """Returns gym space of this observation."""
        return gym.spaces.Box(-INF, INF, (self.env.num_waypoints, 3),
                              dtype=np.float32)

    def get_observation(self):
        """Returns the observation data of the current step."""
        return np.array(self.env.waypoints, dtype=np.float32)


class PolylineEnv(robot_env.RobotEnv):
    """The environment of flying hammer-shape tool."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=False):
        """Initialize."""
        self.config = config or self.default_config
        self.num_modes = self.config.ACTION.NUM_MODES
        self.num_waypoints = self.config.ACTION.NUM_WAYPOINTS

        observations = [
            StateObs(name='state'),
        ]

        reward_fns = []

        super(PolylineEnv, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=simulator,
            config=self.config,
            debug=debug)

        self.configuration_space = gym.spaces.Box(
            low=np.array(self.config.OBSERVATION.LOW),
            high=np.array(self.config.OBSERVATION.HIGH),
            dtype=np.float32)

        action_shape = [self.num_waypoints * 3]
        self.action_space = gym.spaces.Dict({
            'action': gym.spaces.Box(
                low=-np.ones(action_shape, dtype=np.float32),
                high=np.ones(action_shape, dtype=np.float32),
                dtype=np.float32),
            'use_mode': gym.spaces.Discrete(2),
            'mode': gym.spaces.Discrete(self.num_modes),
        })

        self.waypoints = []

    def reset_scene(self):
        """Reset the scene in simulation or the real world."""
        pass

    def reset_robot(self):
        """Reset the robot in simulation or the real world."""
        self.robot_pose = self.configuration_space.sample()
        self.waypoints = self.num_waypoints * [self.robot_pose]
        
        # TODO(debug)
        # self.robot_pose = np.array([0.5, 1.0, 0.0])

    def execute_action(self, action):
        """Execute the robot action.
        """
        action = action['action']
        action = np.reshape(action, [self.num_waypoints, 3])

        if self.debug:
            logger.info('Step: %d; Executing action %r', self.num_steps, action)

        self.waypoints = []

        x = self.robot_pose[0]
        y = self.robot_pose[1]
        theta = self.robot_pose[2]

        for i in range(self.num_waypoints):
            delta_x = action[i, 0] * self.config.ACTION.POSITION_STEP
            delta_y = action[i, 1] * self.config.ACTION.POSITION_STEP
            delta_theta = action[i, 2] * self.config.ACTION.EULER_STEP

            theta = (theta + delta_theta) % (2 * np.pi)
            x = x + delta_x * np.cos(theta)
            y = y + delta_y * np.sin(theta)

            x = np.clip(x,
                        self.configuration_space.low[0],
                        self.configuration_space.high[0])
            y = np.clip(y,
                        self.configuration_space.low[1],
                        self.configuration_space.high[1])
            theta = np.clip((theta + np.pi) % (2 * np.pi) - np.pi,
                            self.configuration_space.low[2],
                            self.configuration_space.high[2])

            waypoint = np.array([x, y, theta])
            self.waypoints.append(waypoint)

        if self.debug:
            print('Waypoints:')
            for i in range(self.num_waypoints):
                waypoint = self.waypoints[i]
                print('%s' % (waypoint))
            self.visualize_waypoints(self.waypoints)

        self.robot_pose = self.waypoints[-1]

    def visualize_waypoints(self, waypoints):
        plt.figure(figsize=(5, 5))

        plt.xlim([self.config.OBSERVATION.LOW[0],
                  self.config.OBSERVATION.HIGH[0]])
        plt.ylim([self.config.OBSERVATION.LOW[1],
                  self.config.OBSERVATION.HIGH[1]])

        for i in range(self.num_waypoints):
            waypoint = waypoints[i]
            plt.scatter(waypoint[0], waypoint[1], c='b', s=50)
            arrow_head = (waypoint[0] + 0.02 * np.cos(waypoint[2]),
                          waypoint[1] + 0.02 * np.sin(waypoint[2]))
            plt.plot([waypoint[0], arrow_head[0]],
                     [waypoint[1], arrow_head[1]],
                     c='b')
            plt.text(arrow_head[0], arrow_head[1], '%d' % (i))

        for i in range(self.num_waypoints):
            if i == 0:
                plt.plot([self.robot_pose[0], waypoints[0][0]],
                         [self.robot_pose[1], waypoints[0][1]],
                         c='g')
            else:
                plt.plot([waypoints[i - 1][0], waypoints[i][0]],
                         [waypoints[i - 1][1], waypoints[i][1]],
                         c='g')

        plt.show()
