"""Grasp and hammer environment for the Sawyer robot.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np

from robovat.math import Pose
from robovat.math import get_transform
from robovat.utils.logging import logger

from envs.tool_env import ToolEnv


class ToolHammerEnv(ToolEnv):
    """Grasp and hammer environment.
    """

    # Paths to objects.
    SLOT_PATH = os.path.join('envs', 'tool_hammer', 'slot.urdf')
    PEG_PATH = os.path.join('envs', 'tool_hammer', 'peg_easy.urdf')

    # Region to load the target object.
    TARGET_REGION = {
            'x': (0.3, 0.5),
            'y': (-0.2, 0.0),
            'z': (0.07, 0.17),
            'roll': 0,
            'pitch': 0,
            'yaw': 0,
            }
    SLOT_OFFSET = 0.1

    # Control routine.
    HAMMER_HEIGHT_OFFSET = 0.115

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

        self.target = None

    def _reset_task(self):
        """Reset the task region.
        """

        # Sample and load a target object.
        if self.is_simulation:
            pose = Pose.uniform(**self.TARGET_REGION)
            peg_pose = get_transform(source=self.table_pose).transform(pose)
            slot_offset = np.array([0, self.SLOT_OFFSET, 0])
            slot_pose = Pose(peg_pose.position + slot_offset,
                             peg_pose.orientation)
            self.slot = self._add_body(self.SLOT_PATH, slot_pose,
                                       is_static=True)
            self.target = self._add_body(self.PEG_PATH, peg_pose)
            # Visualize the goal.
            if self.debug:
                import pybullet

                target = self.target.pose.position

                pybullet.addUserDebugLine(
                        [target[0] - 0.02, target[1], target[2]],
                        [target[0] + 0.02, target[1], target[2]],
                        lineColorRGB=[1, 1, 0],
                        lineWidth=5)

                pybullet.addUserDebugLine(
                        [target[0], target[1] - 0.02, target[2]],
                        [target[0], target[1] + 0.02, target[2]],
                        lineColorRGB=[1, 1, 0],
                        lineWidth=5)
        else:
            hammer_cfg = self.cfg['tool_hammer']
            x = float(hammer_cfg['peg_x'])
            y = float(hammer_cfg['peg_y'])
            z = float(hammer_cfg['peg_z'])
            self.target_pose = Pose([x, y, z], [0, 0, 0])

    def _get_observation(self):
        """Get the observation.

        Returns:
            The dictionary of observations.
        """
        camera_images = self.camera.frames()

        if self.is_simulation:
            graspable_pose = self.graspable.pose.toarray()
            target_pose = self.target.pose
        else:
            if self.graspable_pose is not None:
                graspable_pose = self.graspable_pose.toarray()
            else:
                graspable_pose = None
            target_pose = self.target_pose

        return {
                'depth': camera_images['depth'],
                'graspable_pose': graspable_pose,
                'target_pose': target_pose.toarray(),
                }

    def _task_routine(self, action):
        """Perform the task routine.

        Args:
            action: A 7-dimentional vector [x, y, z, angle, dx, dy, dz].

        Returns:
            See the parent class.
        """
        x = action['task'][0]
        y = action['task'][1]
        if self.is_simulation:
            z = self.HAMMER_HEIGHT + self.table_pose.position[2]
            target_pose = self.target.pose
        else:
            z = action['task'][2]
            target_pose = self.target_pose

        x = action['task'][0] + target_pose.position[0] - 0.07
        y = action['task'][1] + target_pose.position[1]
        z = target_pose.position[2] + self.HAMMER_HEIGHT_OFFSET
        angle = action['task'][2] + 0.5 * np.pi
        euler = [0, np.pi, angle]

        prestart = Pose([x - 0.2, y - 0.2, self.SAFE_HEIGHT], euler)
        start = Pose([x, y, z], euler)
        post = Pose([x - 0.2, y - 0.2, self.SAFE_HEIGHT], euler)

        if self.debug:
            self._visualize_hammer(x, y, z, angle)

        # The task routine.
        status = 'initial'
        while(1):
            status, dropped = self._handle_task(
                    status,
                    prestart,
                    start,
                    post)

            if status == 'done':
                break

            if self.is_simulation:
                self.world.step()

                # Check if the tool is dropped before manipulation.
                if dropped:
                    break

                if status in ['initial', 'overhead', 'prestart', 'start']:
                    if self.world.check_contact(self.graspable, self.target):
                        logger.debug('Tool-peg collision before hammering.')
                        return False, 0.0, dropped

                # Check if the peg or the slot are touched unexpectively.
                if self.world.check_contact(self.robot.arm, self.target):
                    logger.debug('The arm contacts the peg.')
                    break
                elif self.world.check_contact(self.robot.arm, self.slot):
                    logger.debug('The arm contacts the slot.')
                    break
                elif self.world.check_contact(self.graspable, self.slot):
                    logger.debug('The tool contacts the slot.')
                    break

        return self._get_task_success(), self._get_task_reward(), dropped

    def _handle_task(self,
                     status,
                     prestart,
                     start,
                     post):
        """Transit and handle the control status.
        """
        old_status = status
        dropped = False

        if status == 'initial':
            if self.robot.is_limb_ready():
                # self.robot.move_to_joint_positions(self.OVERHEAD_POSITIONS)
                status = 'overhead'

        elif status == 'overhead':
            if self.robot.is_limb_ready():
                self.robot.move_to_gripper_pose(prestart)
                status = 'prestart'

        elif status == 'prestart':
            if self.robot.is_limb_ready():
                self.robot.move_to_gripper_pose(start, straight_line=True)
                status = 'hammer'

        elif status == 'hammer':
            if self.robot.is_limb_ready():
                wrist_joint_angle = self.robot.joint_positions['right_j6']
                positions = {'right_j6': wrist_joint_angle - np.pi}
                self.robot.move_to_joint_positions(positions, speed=0.5,
                                                   timeout=2)
                positions = {'right_j6': wrist_joint_angle}
                self.robot.move_to_joint_positions(positions, speed=0.5,
                                                   timeout=2)
                positions = {'right_j6': wrist_joint_angle - np.pi}
                self.robot.move_to_joint_positions(positions, speed=0.5,
                                                   timeout=2)
                positions = {'right_j6': wrist_joint_angle}
                self.robot.move_to_joint_positions(positions, speed=0.5,
                                                   timeout=2)
                status = 'post'

        elif status == 'post':
            if self.robot.is_limb_ready():
                self.robot.move_to_gripper_pose(post, speed=0.4)
                status = 'reset'

        elif status == 'reset':
            if self.robot.is_limb_ready():
                self.robot.move_to_gripper_pose(self.postgrasp_pose)
                status = 'open'

        elif status == 'open':
            if self.robot.is_gripper_ready():
                self.robot.grip(0)
                status = 'end'

        elif status == 'end':
            if self.robot.is_limb_ready():
                status = 'done'
        else:
            raise ValueError('Unrecognized control status: %s' % status)

        if status != old_status:
            if self.is_simulation:
                if old_status in ['initial', 'overhead', 'prestart']:
                    dropped = (not self.world.check_contact(
                                    self.robot.arm, self.graspable))
            else:
                dropped = None

            logger.debug('Status: %s', status)

        return status, dropped

    def _get_task_reward(self):
        """Get the reward of the task.
        """
        if self.is_simulation:
            if self.world.check_contact(self.target, self.slot):
                initial_position = self.initial_target_pose.position
                current_position = self.target.pose.position
                delta = current_position - initial_position
                distance = delta[1]
                return distance
            else:
                return 0
        else:
            return float(raw_input('Please input reward:'))

    def _get_task_success(self):
        """Get the success of the task.
        """
        if self.is_simulation:
            if self.world.check_contact(self.target, self.slot):
                initial_position = self.initial_target_pose.position
                current_position = self.target.pose.position
                delta = current_position - initial_position
                distance = delta[1]
                return distance > 0.02
            else:
                return False
        else:
            return self.yes_no('Was task succesful?')

    def _visualize_hammer(self, x, y, z, angle):
        """Visualize the task phase.
        """
        if self.is_simulation:
            import pybullet

            angle1 = angle
            angle2 = angle + np.pi / 2
            offset1 = -np.array([np.cos(angle1), np.sin(angle1), 0]) * 0.3
            offset2 = -np.array([np.cos(angle2), np.sin(angle2), 0]) * 0.3

            # Visualize the trajectory.
            pybullet.addUserDebugLine(
                    [x, y, z],
                    [x + offset1[0], y + offset1[1], z],
                    lineColorRGB=[0, 0, 1],
                    lineWidth=2)

            # Visualize the gripper angle.
            pybullet.addUserDebugLine(
                    [x, y, z],
                    [x + offset2[0], y + offset2[1], z],
                    lineColorRGB=[0, 0, 1],
                    lineWidth=2)
