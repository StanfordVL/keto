"""The basic class of robots."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os

from robovat.utils.string_utils import camelcase_to_snakecase
from robovat.utils.yaml_config import YamlConfig


class Robot(object):
    """Base class for robots."""
    __metaclass__ = abc.ABCMeta

    @property
    def pose(self):
        raise NotImplementedError

    @property
    def position(self):
        return self.pose.position

    @property
    def orientation(self):
        return self.pose.orientation

    @property
    def euler(self):
        return self.orientation.euler

    @property
    def quaternion(self):
        return self.orientation.quaternion

    @property
    def matrix3(self):
        return self.orientation.matrix3

    @property
    def default_config(self):
        """Load the default configuration file."""
        env_name = camelcase_to_snakecase(type(self).__name__)
        config_path = os.path.join('configs', 'robots', '%s.yaml' % (env_name))
        assert os.path.exists(config_path), (
            'Default configuration file %s does not exist' % (config_path))
        return YamlConfig(config_path).as_easydict()

    @staticmethod
    @abc.abstractmethod
    def build_in_real_world():
        """Build the robot in the real world.

        Returns:
            The robot in the real world.
        """
        raise NotImplementedError('build_in_real_world() is not implemented'
                                  'for Robot.')

    @staticmethod
    @abc.abstractmethod
    def build_in_simulation(world, pose, is_static):
        """Build the robot in simulation.

        Args:
            world: The simulated world for the robot.
            pose: The initial pose of the robot base.
            is_static: If true, set the robot base to be static.

        Returns:
            The robot in simulation.
        """
        raise NotImplementedError('build_in_simulation() is not implemented'
                                  'for Robot.')
