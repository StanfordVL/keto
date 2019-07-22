"""Tool using environment for the Sawyer robot.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.utils.logging import logger
from robovat.grasp import visualize
from robovat.utils.time_utils import get_timestamp_as_string as get_time

from envs.grasp_env import GraspEnv


class ToolEnv(GraspEnv):
    """Tool using (task-oriented grasping and manipulation) environment.

    Each child class of ToolEnv should implement their own verions of:
        - _reset_task()
        - _get_task_observation()
        - _task_routine()
        - _get_task_reward()
    """

    # Region to load graspable objects.
    GRASPABLE_REGION = {
            'x': -0.3,
            'y': 0.0,
            'z': 0.3,
            'roll': (-np.pi, np.pi),
            'pitch': (0, np.pi),
            'yaw': (-np.pi, np.pi),
            }

    # Region to perform the manipulation task.
    ACTION_SPACE = {
            'x': (0, 0.15),
            'y': (-0.15, 0.15),
            'angle': (0.5 * np.pi, 1.5 * np.pi),
            }

    def __init__(self,
                 is_simulation,
                 data_dir='',
                 key=0,
                 cfg=None,
                 debug=False,
                 camera=None):
        """Initialize.

        See the parent class.
        """
        GraspEnv.__init__(
                 self,
                 is_simulation=is_simulation,
                 data_dir=data_dir,
                 key=key,
                 cfg=cfg,
                 debug=debug,
                 camera=camera,
                 drop=False)  # Dont drop tool!

        self.num_task_successes = 0
        self.num_dropped = 0

    def reset(self):
        """Reset the environment.

        Returns:
        The observation.
        """
        self._reset_robot_and_scene()
        self.robot.reset(self.OUT_OF_VIEW_POSITIONS)

        if self.is_simulation:
            self._clear_scene()

            self._reset_grasp()
            self._reset_task()

            self._wait_until_stable(self.graspable)
            self.initial_graspable_height = self.graspable.position[2]
            self.table_surface_height = self.table_pose.position[2]
        else:
            self._reset_grasp()
            self._reset_task()
            self.table_surface_height = 0.0

        return self._get_observation()

    def step(self, action):
        """Take a step.

        Args:
            action: A 8-dimentional vector.

        Returns:
            See parent class.
        """
        #
        # Grasp phase.
        #
        self.step_start_time = get_time()
        visualize.name_prepend = self.step_start_time
        grasp_success = self._grasp_routine(action)

        # Accumulate and log statistics.
        if grasp_success:
            self.num_grasp_successes += 1
        grasp_success_rate = (float(self.num_grasp_successes) /
                              float(self.num_episodes))
        logger.info('grasp_success: %r, grasp_success_rate: %.4f.',
                    grasp_success, grasp_success_rate)

        #
        # Task phase.
        #

        if grasp_success:
            task_success, task_reward, dropped = self._task_routine(action)
        else:
            task_success = False
            task_reward = 0.0
            dropped = False

        if dropped:
            logger.info('The tool is dropped.')
            self.num_dropped += 1

        dropped_rate = (float(self.num_dropped) /
                        float(self.num_episodes))
        logger.info('dropped_rate: %.4f', dropped_rate)

        # Accumulate and log statistics.
        if task_success:
            self.num_task_successes += 1
        task_success_rate = (float(self.num_task_successes) /
                             float(self.num_episodes))
        logger.info('task_success: %r, '
                    'task_success_rate: %.4f, '
                    'task_reward: %f',
                    task_success,
                    task_success_rate,
                    task_reward)

        observation = dict()
        reward = {
                'grasp_success': grasp_success,
                'task_success': task_success,
                'task_reward': task_reward,
                'dropped': dropped,
                }
        done = True
        info = None

        return observation, reward, done, info

    def _reset_task(self):
        """Reset the manipulation region.
        """
        raise NotImplementedError

    def _task_routine(self, action):
        """Perform the task routine.

        Args:
            action: A 7-dimentional vector [x, y, z, angle, dx, dy, dz].

        Returns:
            The task reward as a float.
        """
        raise NotImplementedError

    def _get_task_reward(self):
        """Get the reward of the task.

        Called by _task_routine().

        Returns:
            The task reward as a float.
        """
        raise NotImplementedError
