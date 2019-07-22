"""Top-down grasping environment for the Sawyer robot.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import glob
import random
import threading
import time

import numpy as np

from robovat.grasp import Grasp2D
from robovat.math import Pose
from robovat.math import get_transform
from robovat.utils.logging import logger


from envs.sawyer_env import SawyerEnv


class GraspEnv(SawyerEnv):
    """Top-down grasping environment.
    """

    # The safe height to move the gripper above the table.
    SAFE_HEIGHT = 0.4

    # Path patterns of graspable objects.
    GRASPABLE_PATHS = [
            # os.path.join('tools', 'mpi-grasp', '*', '*.urdf'),
            # os.path.join('tools', 'jhu-it', '*', '*.urdf'),
            # os.path.join('tools', 'dex-net', '*', '*.urdf'),
      
            os.path.join('tools', 'real-*', '*', '*.urdf'),
            # os.path.join('tools', 'procedural', '*', '*', '*.urdf'),
            ]
    GRASPABLE_SCALE_RANGE = (1, 1)

    # Region to load graspable objects.
    GRASPABLE_REGION = {
            'x': -0.25,
            'y': 0.0,
            'z': 0.4,
            'roll': (-np.pi, np.pi),
            'pitch': (0, np.pi),
            'yaw': (-np.pi, np.pi),
            }

    # Control routinue.
    MIN_GRASP_HEIGHT = 0.15
    GRASP_HEIGHT_OFFSET = 0.125
    MAX_GRASP_STEPS = 4000

    def __init__(self,
                 is_simulation,
                 data_dir='',
                 key=0,
                 cfg=None,
                 debug=False,
                 camera=None,
                 drop=True):
        """Initialize.

        See the parent class.
        """
        SawyerEnv.__init__(
                 self,
                 is_simulation=is_simulation,
                 data_dir=data_dir,
                 key=key,
                 cfg=cfg,
                 debug=debug,
                 camera=camera)

        # Instance var for whether to drop after grasping
        # Meant to possibly be set to other values in subclasses
        self.drop = drop

        # Find paths to all graspable objects.
        if self.is_simulation:
            self.graspable = None

            self.graspable_paths = []
            for relative_pattern in self.GRASPABLE_PATHS:
                pattern = os.path.join(self.data_dir,'simulation',
                                       relative_pattern)
                paths = [os.path.abspath(path) for path in glob.glob(pattern)]
                self.graspable_paths += paths

            logger.debug('Found %d graspable objects.',
                         len(self.graspable_paths))
        else:
            # Tune constants for real world
            self.GRASP_HEIGHT_OFFSET = 0.07
            self.MIN_GRASP_HEIGHT = 0.12
            self.SAFE_HEIGHT = 0.22

        # Statistics.
        self.num_grasp_successes = 0

        if cfg['capture_continuous']:
            # Whether to output images from throughout episode, even if policy
            # is one shot and there is only one observation
            # TODO would be ideal to have this logic in SawyerEnv and not here
            thread = threading.Thread(target=self._append_to_observation,
                                      args=())
            thread.daemon = True  # Daemonize thread
            thread.start()        # Start the execution

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
            self._wait_until_stable(self.graspable)

            self.initial_graspable_height = self.graspable.position[2]
        else:
            self.graspable_pose = None
        self._observation = self._get_observation()

        return self._observation

    def step(self, action):
        """Take a step.

        See the parent class.
        """
        # Grasp.
        grasp_success = self._grasp_routine(action)

        # Returns.
        observation = self._get_observation()
        reward = float(grasp_success)
        done = True
        info = None

        # Accumulate and log statistics.
        if grasp_success:
            self.num_grasp_successes += 1

        success_rate = float(self.num_grasp_successes) / self.num_episodes
        logger.info('grasp_success: %r, success_rate: %.3f.'
                    % (grasp_success, success_rate))

        return observation, reward, done, info

    def _reset_grasp(self):
        """Reset the grasping region.

        Sample and load a graspable object.
        """
        if self.is_simulation:
            path = random.choice(self.graspable_paths)
            pose = Pose.uniform(**self.GRASPABLE_REGION)
            pose = get_transform(source=self.table_pose).transform(pose)
            scale = np.random.uniform(*self.GRASPABLE_SCALE_RANGE)
            logger.info('Loading the grasp object from %s with scale %.2f.',
                        path, scale)
            self.graspable = self._add_body(path, pose, scale=scale)
            self.graspable = self._add_body(path, pose, scale=scale)
        else:
            self.graspable_pose = None

    def _get_observation(self):
        """Get the observation.

        Returns:
            The dictionary of observations.
        """
        camera_images = self.camera.frames()

        if self.is_simulation:
            graspable_pose = self.graspable.pose
            graspable_name = self.graspable.name,
        else:
            graspable_pose = self.graspable_pose
            graspable_name = 'grasping'

        return {
                'depth': camera_images['depth'],
                'graspable_name': graspable_name,
                'graspable_pose': graspable_pose
                }

    def _grasp_routine(self, action):
        """Take a grasping routine.

        Args:
            action: A 4-dimentional vector represents [x, y, z, angle].

        Returns:
            grasp_success: True or False.
        """
        # Convert the action into grasp poses.
        grasp = Grasp2D.from_vector(action['grasp'], camera=self.camera)
        x, y, z, angle = grasp.action
        angle %= 2 * np.pi

        grasp_height = max(self.GRASP_HEIGHT_OFFSET + z, self.MIN_GRASP_HEIGHT)
        if self.is_simulation:
            grasp_height += self.table_pose.position[2]
        else:
            x += self.cfg['grasp']['offset_x']
            y += self.cfg['grasp']['offset_y']
        euler = [0, np.pi, angle]

        # Prepare the grasp action.
        pregrasp_pose = Pose([x, y, self.SAFE_HEIGHT], euler)
        grasp_pose = Pose([x, y, grasp_height], euler)
        postgrasp_pose = Pose([x, y, self.SAFE_HEIGHT], euler)
        self.postgrasp_pose = postgrasp_pose

        if self.debug:
            self.visualize_grasp(x, y, z, angle)

        # The grasp routine.
        status = 'initial'
        grasp_steps = 0
        while(1):
            status = self._handle_grasp(status,
                                        pregrasp_pose,
                                        grasp_pose,
                                        postgrasp_pose)

            if self.is_simulation:
                self.world.step()

                if status == 'grasp':
                    if grasp_steps >= self.MAX_GRASP_STEPS:
                        return False
                    grasp_steps += 1

            if status == 'done':
                break

        # Check grasp success.
        if self.is_simulation:
            self._wait_until_stable(self.graspable, 0.1, None)

        return self._check_grasp_success()

    def _append_to_observation(self):
        """Add to current observation
        """
        if self._observation is not None:
            if 'continuous' not in self._observation:
                self._observation['continuous'] = []
            camera_images = self.camera.frames()
            self._observation['continuous'].append(camera_images)
            time.sleep(self.cfg['capture_continuous_freq'])

    def _handle_grasp(self,
                      status,
                      pregrasp_pose,
                      grasp_pose,
                      postgrasp_pose):
        """Handle the grasping action.
        """
        old_status = status

        if status == 'initial':
            if self.robot.is_limb_ready():
                status = 'overhead'
                if self.is_simulation:
                    self.robot.move_to_joint_positions(self.OVERHEAD_POSITIONS)

        elif status == 'overhead':
            if self.robot.is_limb_ready():
                status = 'pregrasp'
                self.robot.move_to_gripper_pose(pregrasp_pose)

        elif status == 'pregrasp':
            if self.robot.is_limb_ready():
                status = 'grasp'
                self.robot.move_to_gripper_pose(grasp_pose, speed=0.075)

        elif status == 'grasp':
            if self.robot.is_limb_ready():
                status = 'close_gripper'
                self.robot.grip(1)

        elif status == 'close_gripper':
            if self.robot.is_gripper_ready():
                status = 'postgrasp'
                self.robot.move_to_gripper_pose(postgrasp_pose)

        elif status == 'postgrasp':
            if self.robot.is_limb_ready() and self.robot.is_gripper_ready():
                self.robot.move_to_gripper_pose(postgrasp_pose, speed=0.075)
                status = 'open' if self.drop else 'done'

        elif status == 'open':
            if self.robot.is_limb_ready():
                self.robot.grip(0)
                status = 'done'

        else:
            raise ValueError('Unrecognized control status: %s' % status)

        if status != old_status:
            logger.debug('Status: %s', status)
            if self.take_images:
                self.take_image(status)

        return status

    def _check_grasp_success(self):
        """Check if the grasp is successful or not.
        """
        if self.is_simulation:
            success = self.world.check_contact(self.robot.arm, self.graspable)
        else:
            if self.cfg['grasp']['ask_grasp_success']:
                success = self.yes_no('Did grasp succeed?')
            else:
                success = True

        return success

    def visualize_grasp(self, x, y, z, angle):
        """Visualize the action.

        See the parent class.
        """
        angle -= np.pi / 2
        grasp_pose = [[x, y, z], [0, np.pi, angle]]
        grasp_pose = Pose(grasp_pose[0], grasp_pose[1])

        grasp_center = grasp_pose.position
        offset = np.array([np.cos(angle), np.sin(angle), 0]) * 0.1
        grasp_start = grasp_center - offset
        grasp_end = grasp_center + offset

        ee_offset = np.array([0, 0, self.GRASP_HEIGHT_OFFSET])
        ee_center = grasp_center + ee_offset
        ee_start = ee_center - offset
        ee_end = ee_center + offset

        if self.is_simulation:
            import pybullet

            # Visualize the grasp.
            pybullet.addUserDebugLine(
                    grasp_start,
                    grasp_end,
                    lineColorRGB=[1, 0, 1],
                    lineWidth=6)

            # Visualize the end enffector link pose.
            pybullet.addUserDebugLine(
                    ee_start,
                    ee_end,
                    lineColorRGB=[1, 0, 1],
                    lineWidth=2)

            pybullet.addUserDebugLine(
                    grasp_center,
                    ee_center,
                    lineColorRGB=[1, 0, 1],
                    lineWidth=2)

            # Visualize the ray from the camera to the grasp center.
            pybullet.addUserDebugLine(
                    self.camera.pose.position,
                    grasp_pose.position,
                    lineColorRGB=[1, .8, 1],
                    lineWidth=2)

    def yes_no(self, answer):
        """ Little helper method for checking success via terminal """
        yes = set(['yes', 'y', 'ye', ''])
        no = set(['no', 'n'])

        while True:
            choice = raw_input(answer).lower()
            if choice in yes:
                return True
            elif choice in no:
                return False
            else:
                print("Please respond with 'yes' or 'no'\n")
