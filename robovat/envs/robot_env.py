"""The parent class for robot environments.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import os.path

import gym
import gym.spaces

from robovat.utils.string_utils import camelcase_to_snakecase
from robovat.utils.yaml_config import YamlConfig


class RobotEnv(gym.Env):
    """The parent class for robot environments."""

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 observations,
                 reward_fns,
                 simulator=None,
                 config=None,
                 debug=False):
        """Initialize.
        """
        self.simulator = simulator
        self.config = config or self.default_config
        self.debug = debug

        self._num_episodes = -1
        self._num_steps = 0
        self._episode_reward = 0.0
        self._is_done = True

        self.observations = observations
        self.reward_fns = reward_fns

        for obs in self.observations:
            obs.initialize(self)
        for reward_fn in self.reward_fns:
            reward_fn.initialize(self)

        self.observation_space = gym.spaces.Dict([
            (obs.name, obs.get_gym_space()) for obs in self.observations])

        self.action_space = None

    @property
    def default_config(self):
        """Load the default configuration file."""
        env_name = camelcase_to_snakecase(type(self).__name__)
        config_path = os.path.join('configs', 'envs', '%s.yaml' % (env_name))
        assert os.path.exists(config_path), (
            'Default configuration file %s does not exist' % (config_path))
        return YamlConfig(config_path).as_easydict()

    @property
    def info(self):
        """Information of the environment."""
        return {'name': type(self).__name__}

    @property
    def num_episodes(self):
        """Number of episodes."""
        return self._num_episodess

    @property
    def num_steps(self):
        """Number of steps of the current episode."""
        return self._num_steps

    @property
    def episode_reward(self):
        """Received reward of the current episode."""
        return self._episode_reward

    @property
    def is_done(self):
        """If the episode is done."""
        return self._is_done

    def reset(self):
        """Reset."""
        if not self._is_done and self._num_steps > 0:
            for obs in self.observations:
                obs.on_episode_end()
            for reward_fn in self.reward_fns:
                reward_fn.on_episode_end()

        self._num_episodes += 1
        self._num_steps = 0
        self._episode_reward = 0.0
        self._is_done = False

        if self.simulator:
            self.simulator.reset()
            self.simulator.start()

        self.reset_scene()
        self.reset_robot()

        for obs in self.observations:
            obs.on_episode_start()
        for reward_fn in self.reward_fns:
            reward_fn.on_episode_start()

        return self.get_observation()

    def step(self, action):
        """Take a step."""
        if not self.action_space.contains(action):
            raise ValueError('Invalid action: %r' % (action))

        if self._is_done:
            raise ValueError('The environment is done. Forget to reset?')

        self.execute_action(action)

        observation = self.get_observation()

        reward = 0.0

        for reward_fn in self.reward_fns:
            reward_value, termination = reward_fn.get_reward()
            reward += reward_value

            if termination:
                self._is_done = True

        self._episode_reward += reward
        self._num_steps += 1
        
        return observation, reward, self._is_done, None

    def get_observation(self):
        """Return the observation."""
        observation = collections.OrderedDict()

        for obs in self.observations:
            obs_data = obs.get_observation()
            observation[obs.name] = obs_data

        return observation

    @abc.abstractmethod
    def reset_scene(self):
        """Reset the scene in simulation or the real world.
        """
        pass

    @abc.abstractmethod
    def reset_robot(self):
        """Reset the robot in simulation or the real world.
        """
        pass

    @abc.abstractmethod
    def execute_action(self, action):
        """Execute the robot action.
        """
        pass
