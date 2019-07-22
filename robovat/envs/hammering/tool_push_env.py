"""Grasp and push environment for the Sawyer robot.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import random

import numpy as np

from robovat.math import Pose
from robovat.math import get_transform
from robovat.utils.logging import logger

from envs.tool_env import ToolEnv


class ToolPushEnv(ToolEnv):
    """Grasp and push environment.
    """

    # Path patterns of graspable objects.
    BOWL_PATH = os.path.join('envs', 'tool_push', 'bowl', 'bowl.urdf')
    TARGET_PATHS = [
            # os.path.join('envs', 'tool_push', 'cube.urdf'),
            os.path.join('envs', 'tool_push', 'cylinder.urdf'),
            ]
    TARGET_SCALE_RANGE = (0.8, 1.0)

    # Region to load the target object.
    TARGET_REGION = {
            'x': (0.15, 0.25),
            'y': (-0.25, -0.20),
            'z': 0.1,
            'roll': 0,
            'pitch': 0,
            'yaw': (-np.pi, np.pi),
            }

    # Region to sample the goal.
    GOAL_REGION = {
            'x': (0.2, 0.2),
            'y': -0.4,
            'z': -0.1,
            }

    # Control routine.
    PUSH_HEIGHT = 0.19
    MAX_PRESTART_STEPS = 1000
    MAX_PUSH_STEPS = 4000

    # The threshold for checking pushing success.
    TARGET_MIN_HEIGHT = 0.2

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
        ToolEnv.__init__(
                 self,
                 is_simulation=is_simulation,
                 data_dir=data_dir,
                 key=key,
                 cfg=cfg,
                 debug=debug,
                 camera=camera)

        # Target object.
        self.target = None
        self.goal = None
        if not self.is_simulation:
            self.PUSH_HEIGHT = 0.15

    def _reset_task(self):
        """Reset the task region.
        """
        if self.is_simulation:
            # Sample and load a target object.
            path = random.choice(self.TARGET_PATHS)
            scale = np.random.uniform(*self.TARGET_SCALE_RANGE)
            pose = Pose.uniform(**self.TARGET_REGION)
            pose = get_transform(source=self.table_pose).transform(pose)
            self.target = self._add_body(path, pose, scale=scale)

            pose = Pose.uniform(**self.GOAL_REGION)
            pose = get_transform(source=self.table_pose).transform(pose)
            self.goal = np.array(pose.position[0:2])
            self.bowl = self._add_body(self.BOWL_PATH, pose, is_static=True)

            # Visualize the goal.
            if self.debug:
                import pybullet

                pose = Pose.uniform(**self.GOAL_REGION)
                pose = get_transform(source=self.table_pose).transform(pose)
                self.goal = np.array(pose.position[0:2])
                self.goal_object = self._add_body(self.BOWL_PATH, pose,
                                                  is_static=True)

                pybullet.addUserDebugLine(
                        [self.goal[0], self.goal[1] - 0.02, 0],
                        [self.goal[0], self.goal[1] + 0.02, 0],
                        lineColorRGB=[1, 1, 0],
                        lineWidth=5)
        else:
            self.target_pose = self.camera.get_pose_with_click(
                               "Click on the object to be pushed")
            self.goal = self.camera.get_pose_with_click(
                               "Click on the location to push to").position

    def _task_routine(self, action):
        """Perform the task routine.

        Args:
            action: A 7-dimentional vector [x, y, z, angle, dx, dy, dz].

        Returns:
            See the parent class.
        """

        assert (action['task'][2] >= -0.5 * np.pi and
                action['task'][2] <= 0.5 * np.pi)

        if self.is_simulation:
            self.initial_target_pose = self.target.pose
            target_pose = self.target.pose
        else:
            self.initial_target_pose = self.target_pose
            target_pose = self.target_pose

        dist_x = target_pose.position[0] - self.goal[0]
        dist_y = target_pose.position[1] - self.goal[1]
        dist = np.linalg.norm([dist_x, dist_y])
        push_angle = np.arctan2(dist_y, dist_x) % (2 * np.pi)
        x0 = self.goal[0] + (dist + 0.2) * np.cos(push_angle)
        y0 = self.goal[1] + (dist + 0.2) * np.sin(push_angle)

        x = action['task'][0] + x0
        y = action['task'][1] + y0
        if self.is_simulation:
            z = self.PUSH_HEIGHT + self.table_surface_height
        else:
            z = self.PUSH_HEIGHT
        z = self.PUSH_HEIGHT + self.table_surface_height
        angle = action['task'][2] + push_angle
        euler = [0, np.pi, angle]

        start_to_goal = np.linalg.norm([self.goal[0] - x, self.goal[1] - y])
        end_to_goal = 0.1
        push_dist = max(start_to_goal - end_to_goal, 0.0)
        dx = - push_dist * np.cos(push_angle)
        dy = - push_dist * np.sin(push_angle)
        euler = [0, np.pi, angle]

        prestart = Pose([x, y, self.SAFE_HEIGHT], euler)
        start = Pose([x, y, z], euler)
        end = Pose([x + dx, y + dy, z], euler)

        if self.debug and self.is_simulation:
            self._visualize_push(x, y, z, push_angle, dx, dy, 0)

        # The task routine.
        status = 'initial'
        self.prestart_steps = 0
        self.task_steps = 0
        while(1):
            status, dropped = self._handle_push(
                    status,
                    prestart,
                    start,
                    end)

            if status == 'done':
                break

            if self.is_simulation:
                self.world.step()

                # Check if the tool is dropped before manipulation.
                if dropped:
                    break

                # Check if the pushing is stuck.
                if status == 'prestart':
                    self.prestart_steps += 1

                # Check if the tool hits the target moving to start point.
                if status == 'start':
                    if self.world.check_contact(self.graspable, self.target):
                        logger.debug('The tool collides with the target.')
                        break

                # Check if the pushing is stuck.
                if status == 'push':
                    self.task_steps += 1
                    if self.task_steps > self.MAX_PUSH_STEPS:
                        logger.debug('Timeout for pushing.')
                        break
                    if self.world.check_contact(self.target, self.bowl):
                        break
                    if self.world.check_contact(self.target, self.ground):
                        logger.debug('The target hits the ground.')
                        break

                if self.is_simulation and self.world.check_contact(
                                          self.robot.arm, self.target):
                    logger.debug('The arm contacts the target.')
                    break

        return self._get_task_success(), self._get_task_reward(), dropped

    def _handle_push(self,
                     status,
                     prestart,
                     start,
                     end):
        """Transit and handle the control status.
        """
        old_status = status
        dropped = False

        if status == 'initial':
            if self.robot.is_limb_ready():
                status = 'overhead'

        elif status == 'overhead':
            if self.robot.is_limb_ready():
                status = 'prestart'
                self.robot.move_to_gripper_pose(prestart)

        elif status == 'prestart':
            if (self.robot.is_limb_ready() or
                    self.prestart_steps > self.MAX_PRESTART_STEPS):
                status = 'start'
                self.robot.move_to_gripper_pose(start, straight_line=False)

        elif status == 'start':
            if self.robot.is_limb_ready():
                self.robot.move_to_gripper_pose(end, straight_line=True,
                                                speed=0.15)
                status = 'reset'

        elif status == 'reset':
            if self.robot.is_limb_ready() and self.robot.is_gripper_ready():
                self.robot.move_to_gripper_pose(self.postgrasp_pose)
                status = 'open'

        elif status == 'open':
            if self.robot.is_limb_ready():
                self.robot.grip(0)
                status = 'push'

        elif status == 'push':
            if self.robot.is_limb_ready():
                status = 'done'

        else:
            raise ValueError('Unrecognized control status: %s' % status)

        if status != old_status:
            if self.is_simulation:
                if old_status in ['initial', 'overhead', 'prestart']:
                    dropped = (
                            # Tool dropped.
                            (not self.world.check_contact(
                                    self.robot.arm, self.graspable)) or
                            # Target dropped.
                            (not self.world.check_contact(
                                    self.target, self.table))
                            )
            else:
                if self.take_images:
                    self.take_image(status)
                dropped = None

            logger.debug('Status: %s', status)

        return status, dropped

    def _get_observation(self):
        """Get the observation.

        Returns:
            The dictionary of observations.
        """
        camera_images = self.camera.frames()

        if self.is_simulation:
            graspable_pose = self.graspable.pose.toarray()
            target_pose = self.target.pose.toarray()
            goal = self.goal
        else:
            graspable_pose = None
            target_pose = self.target_pose.toarray()
            goal = self.goal

        return {
                'depth': camera_images['depth'],

                # Only available in simulation:
                'graspable_pose': graspable_pose,
                'target_pose': target_pose,
                'goal': goal,
                }

    def _get_task_reward(self):
        """Get the reward of the task.
        """
        if not self.is_simulation:
            return float(raw_input('Please provide task reward\n'))
        else:
            if self.world.check_contact(self.target, self.bowl):
                pushed_delta = (self.target.pose.position[0:2] -
                                self.initial_target_pose.position[0:2])
                goal_delta = self.goal - self.initial_target_pose.position[0:2]
                goal_dist = np.linalg.norm(goal_delta)
                goal_direction = goal_delta / goal_dist
                pushed_dist_towards_goal = np.dot(pushed_delta, goal_direction)
                return np.clip(pushed_dist_towards_goal, 0.0, goal_dist)
            else:
                return 0.0

    def _get_task_success(self):
        """Get the success of the task.
        """
        if not self.is_simulation:
            return self.yes_no('Was task succesful?')
        else:
            # Success if the target falls off the table in bowl.
            if self.world.check_contact(self.target, self.bowl):
                return True
            else:
                return False

    def _visualize_push(self, x, y, z, angle, dx, dy, dz):
        """Visualize the task phase.
        """
        if self.is_simulation:
            import pybullet

            _angle = angle + np.pi / 2
            offset = np.array([np.cos(_angle), np.sin(_angle), 0]) * 0.1

            # Visualize the trajectory.
            pybullet.addUserDebugLine(
                    [x, y, z],
                    [x + dx, y + dy, z],
                    lineColorRGB=[0, 0, 1],
                    lineWidth=2)

            # Visualize the gripper angle.
            pybullet.addUserDebugLine(
                    [x - offset[0], y - offset[1], z],
                    [x + offset[0], y + offset[1], z],
                    lineColorRGB=[0, 0, 1],
                    lineWidth=2)
