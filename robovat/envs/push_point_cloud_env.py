
import glob
import random

import gym
import numpy as np

import pybullet
import time

from robovat.envs import arm_env
from robovat.envs.observations import camera_obs
from robovat.envs.reward_fns.grasp_reward import GraspReward
from robovat.envs.reward_fns.push_reward import PushReward
from robovat.grasp import Grasp2D
from robovat.math import Pose
from robovat.math import get_transform
from robovat.robots import sawyer
from robovat.utils.logging import logger

import matplotlib.pyplot as plt

GRASPABLE_NAME = 'graspable'


class PushPointCloudEnv(arm_env.PushArmEnv):
    """Top-down 4-DoF grasping environment."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=True,
                 is_training=True):
        """Initialize."""
        self.simulator = simulator
        self.config = config or self.default_config
        self.debug = debug
        self.is_training = is_training

        self.build_camera(
            height=self.config.KINECT2.DEPTH.HEIGHT,
            width=self.config.KINECT2.DEPTH.WIDTH,
            intrinsics=self.config.KINECT2.DEPTH.INTRINSICS,
            translation=self.config.KINECT2.DEPTH.TRANSLATION,
            rotation=self.config.KINECT2.DEPTH.ROTATION,
            intrinsics_noise=self.config.KINECT2.DEPTH.INTRINSICS_NOISE,
            translation_noise=self.config.KINECT2.DEPTH.TRANSLATION_NOISE,
            rotation_noise=self.config.KINECT2.DEPTH.ROTATION_NOISE,
            calibration_path=None,
            crop=self.config.KINECT2.DEPTH.CROP)

        observations = [
            camera_obs.SegmentedPointCloudObs(
                camera=self.camera,
                num_points=self.config.OBSERVATION.NUM_POINTS,
                body_names=['graspable'],
                name='point_cloud'),
            camera_obs.CameraIntrinsicsObs(
                name='intrinsics',
                camera=self.camera),
            camera_obs.CameraTranslationObs(
                name='translation',
                camera=self.camera),
            camera_obs.CameraRotationObs(
                name='rotation',
                camera=self.camera)
        ]

        reward_fns = [
            GraspReward(
                name='grasp_reward',
                end_effector_name=sawyer.SawyerSim.ARM_NAME,
                graspable_name=GRASPABLE_NAME,
                target_name='target_0'),
            PushReward(
                name='push_reward',
                graspable_name=GRASPABLE_NAME,
                target_name=['target_0', 'target_1'])
        ]

        if self.simulator:
            self.graspable = None
            self.graspable_path = None
            self.graspable_pose = None
            self.all_graspable_paths = []
            self.graspable_index = 0

            for pattern in self.config.SIM.GRASPABLE.PATHS:
                if pattern[-4:] == '.txt':
                    with open(pattern, 'r') as f:
                        paths = [line.rstrip('\n') for line in f]
                else:
                    paths = glob.glob(pattern)

                self.all_graspable_paths += paths

            self.all_graspable_paths.sort()
            num_graspable_paths = len(self.all_graspable_paths)
            assert num_graspable_paths > 0, (
                'Found no graspable objects at %s'
                % (self.config.SIM.GRASPABLE.PATHS))
            logger.debug('Found %d graspable objects.', num_graspable_paths)

        super(PushPointCloudEnv, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=self.simulator,
            config=self.config,
            debug=self.debug)

        low = np.array(self.config.ACTION.GRASP.LOW + [0.0])
        high = np.array(self.config.ACTION.GRASP.HIGH + [2 * np.pi])
        space_grasp = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32)

        num_steps = self.config.ACTION.TASK.T
        low = np.array(self.config.ACTION.TASK.LOW)
        high = np.array(self.config.ACTION.TASK.HIGH)
        low_task = np.tile(low[np.newaxis, :], [num_steps, 1])
        high_task = np.tile(high[np.newaxis, :], [num_steps, 1])
        space_task = gym.spaces.Box(
            low=low_task,
            high=high_task,
            dtype=np.float32)

        num_keypoints = self.config.ACTION.KEYPOINTS.NUM
        low = np.array(self.config.ACTION.TASK.LOW)
        high = np.array(self.config.ACTION.TASK.HIGH)
        low_keypoints = np.tile(low[np.newaxis, :], [num_keypoints, 1])
        high_keypoints = np.tile(high[np.newaxis, :], [num_keypoints, 1])
        space_keypoints = gym.spaces.Box(
            low=low_keypoints,
            high=high_keypoints,
            dtype=np.float32) 

        self.action_space = gym.spaces.Dict(
            {'grasp': space_grasp,
             'task': space_task,
             'keypoints': space_keypoints})

    def reset_scene(self):
        """Reset the scene in simulation or the real world.
        """
        super(PushPointCloudEnv, self).reset_scene()

        # Reload graspable object.
        if self.config.SIM.GRASPABLE.RESAMPLE_N_EPISODES:
            if (self.num_episodes %
                    self.config.SIM.GRASPABLE.RESAMPLE_N_EPISODES == 0):
                self.graspable_path = None

        if self.graspable_path is None:
            if self.config.SIM.GRASPABLE.USE_RANDOM_SAMPLE:
                self.graspable_path = random.choice(
                    self.all_graspable_paths)
            else:
                self.graspable_index = ((self.graspable_index + 1) %
                                        len(self.all_graspable_paths))
                self.graspable_path = (
                    self.all_graspable_paths[self.graspable_index])

        pose = Pose.uniform(x=self.config.SIM.GRASPABLE.POSE.X,
                            y=self.config.SIM.GRASPABLE.POSE.Y,
                            z=self.config.SIM.GRASPABLE.POSE.Z,
                            roll=self.config.SIM.GRASPABLE.POSE.ROLL,
                            pitch=self.config.SIM.GRASPABLE.POSE.PITCH,
                            yaw=self.config.SIM.GRASPABLE.POSE.YAW)
        pose = get_transform(source=self.table_pose).transform(pose)
        scale = np.random.uniform(*self.config.SIM.GRASPABLE.SCALE)
        logger.info('Loaded the graspable object from %s with scale %.2f...',
                    self.graspable_path, scale)
        self.graspable = self.simulator.add_body(
            self.graspable_path, pose, scale=scale, name=GRASPABLE_NAME)
        logger.debug('Waiting for graspable objects to be stable...')
        self.simulator.wait_until_stable(self.graspable)

        # Reset camera.
        intrinsics = self.intrinsics
        translation = self.translation
        rotation = self.rotation

        if self.intrinsics_noise:
            intrinsics += np.random.uniform(
                -self.intrinsics_noise, self.intrinsics_noise)

        if self.translation_noise:
            translation += np.random.uniform(
                -self.translation_noise, self.translation_noise)

        if self.rotation_noise:
            rotation += np.random.uniform(
                -self.rotation_noise, self.rotation_noise)

        self.camera.set_calibration(intrinsics, translation, rotation)

    def reset_robot(self):
        """Reset the robot in simulation or the real world.
        """
        super(PushPointCloudEnv, self).reset_robot()
        self.robot.reset(self.config.ARM.OFFSTAGE_POSITIONS)

    def feedback(self, reward):
        return

    def execute_action(self, action):

        action_grasp = action['grasp']
        action_task = action['task']
        #
        # Grasping
        #
        is_good_grasp = self._execute_action_grasping(action_grasp)

        if self.is_training and not is_good_grasp:
            return
        #
        # Pushing
        #
        self._execute_action_pushing(action_task)

    def _execute_action_grasping(self, action):
        """Execute the grasp action.

        Args:
            action: A 4-DoF grasp defined in the image space or the 3D space.
        """
        if self.config.ACTION.TYPE == 'CUBOID':
            x, y, z, angle = action
        elif self.config.ACTION.TYPE == 'IMAGE':
            grasp = Grasp2D.from_vector(action, camera=self.camera)
            x, y, z, angle = grasp.as_4dof()
        else:
            raise ValueError(
                'Unrecognized action type: %r' % (self.config.ACTION.TYPE))

        start = Pose(
            [[x, y, z - self.config.SIM.Z_OFFSET], [0, np.pi, angle]])

        phase = 'initial'

        # Handle the simulation robustness.
        if self.simulator:
            num_action_steps = 0

        while(phase != 'done'):
            if self.simulator:
                self.simulator.step()
                if phase == 'start':
                    num_action_steps += 1

            if self.is_phase_ready(phase, num_action_steps):
                phase = self.get_next_phase(phase)
                logger.debug('grasping phase: %s', phase)

                if phase == 'overhead':
                    self.robot.move_to_joint_positions(
                        self.config.ARM.OVERHEAD_POSITIONS)
                    # self.robot.grip(0)

                elif phase == 'prestart':
                    prestart = start.copy()
                    prestart.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(prestart)

                elif phase == 'start':
                    pre_grasp_pose = np.array(self.graspable.pose.position)
                    pre_grasp_euler = self.graspable.pose.euler

                    self.robot.move_to_gripper_pose(start, straight_line=True)

                    # Prevent problems caused by unrealistic frictions.
                    if self.simulator:
                        self.robot.l_finger_tip.set_dynamics(
                            lateral_friction=0.001,
                            spinning_friction=0.001)
                        self.robot.r_finger_tip.set_dynamics(
                            lateral_friction=0.001,
                            spinning_friction=0.001)
                        self.table.set_dynamics(
                            lateral_friction=100)

                elif phase == 'end':
                    self.robot.grip(1)
                    post_grasp_pose = np.array(self.graspable.pose.position)

                elif phase == 'postend':
                    postend = self.robot.end_effector.pose
                    postend.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(
                        postend,
                        straight_line=True, speed=0.3)
                    post_grasp_euler = self.graspable.pose.euler

                    # Prevent problems caused by unrealistic frictions.
                    if self.simulator:
                        self.robot.l_finger_tip.set_dynamics(
                            lateral_friction=1000,
                            rolling_friction=1000,
                            spinning_friction=1000)
                        self.robot.r_finger_tip.set_dynamics(
                            lateral_friction=1000,
                            rolling_friction=1000,
                            spinning_friction=1000)
                        self.table.set_dynamics(
                            lateral_friction=0.3 if self.is_training else 0.01)
        good_loc = self._good_grasp(pre_grasp_pose, post_grasp_pose)
        good_rot = self._good_grasp(np.sin(pre_grasp_euler - post_grasp_euler),
                0, thres=0.17)
        return good_loc

    def _execute_action_pushing(self, action):
        """Execute the pushing action.
        """
        phase = 'initial'
        if self.simulator:
            num_action_steps = 0

        while(phase != 'done'):
            if self.simulator:
                self.simulator.step()
                if phase == 'start':
                    num_action_steps += 1

            if self.is_phase_ready(phase, num_action_steps):
                phase = self.get_next_phase(phase)
                logger.debug('task phase: %s', phase)

                if phase == 'overhead':
                    pass

                elif phase == 'prestart':
                    # move the tool based on action
                    # self._draw_path(action)
                    num_move_steps = action.shape[0]

                    for step in range(1, num_move_steps):
                        if self._robot_should_stop():
                            break
                        if self.timeout:
                            break
                        x, y, z, angle = action[step]
                        angle = (angle + np.pi) % (np.pi * 2) - np.pi

                        [curr_x, curr_y, curr_z
                         ] = self.robot.end_effector.pose.position
                        curr_rz = (self.robot.end_effector.pose.euler[2] + np.pi
                                   ) % (np.pi * 2) - np.pi
                        gripper_pose = np.array(
                            [curr_x, curr_y, curr_z, curr_rz])

                        logger.debug('current pose {}'.format(
                            gripper_pose))
                        logger.debug('moving to {}'.format(action[step]))
                        logger.debug('pose delta {}'.format(
                            action[step] - gripper_pose))

                        pose = Pose(
                            [[x, y, z],
                             [0, np.pi, angle]])
                        # self.plot_pose(pose, 0.1)
                        self.robot.move_to_gripper_pose(
                            pose, straight_line=True,
                            timeout=2,
                            speed=0.7)
                        ready = False
                        time_start = time.time()
                        while(not ready):
                            if self.timeout:
                                break
                            if time.time() - time_start > 2:
                                if step < num_move_steps - 1:
                                    self.timeout = True
                            if self.simulator:
                                self.simulator.step()
                            ready = self.is_phase_ready(
                                phase, num_action_steps)

                elif phase == 'start':
                    pass 

                elif phase == 'end':
                    pass

                elif phase == 'postend':
                    pass

    def _draw_path(self, action):
        plt.figure(figsize=(4, 3))
        plt.plot(action[:, 0], action[:, 1], c='green')
        plt.xlim((-0.5, 1.5))
        plt.ylim((-0.5, 0.5))
        plt.savefig('./episodes/figures/path_%02d' % (
            np.random.randint(100)))
        plt.close()


    def _robot_should_stop(self):
        if not self.simulator.check_contact(
                self.robot.arm, self.graspable):
            return True
        if self.simulator.check_contact(
                self.table, self.graspable):
            return True

    def _good_grasp(self, pre, post, thres=0.02):
        if not self.is_training:
            return True
        trans = np.linalg.norm(pre - post)
        logger.debug('The tool slips {:.3f}'.format(trans))
        return trans < thres

    def _wait_until_ready(self, phase, num_action_steps):
        ready = False
        while(not ready):
            if self.simulator:
                self.simulator.step()
            ready = self.is_phase_ready(
                phase, num_action_steps)
        return

    def get_next_phase(self, phase):
        """Get the next phase of the current phase.

        Args:
            phase: A string variable.

        Returns:
            The next phase as a string variable.
        """
        phase_list = ['initial',
                      'overhead',
                      'prestart',
                      'start',
                      'end',
                      'postend',
                      'done']

        if phase in phase_list:
            i = phase_list.index(phase)
            if i == len(phase_list):
                raise ValueError('phase %r does not have a next phase.')
            else:
                return phase_list[i + 1]
        else:
            raise ValueError('Unrecognized phase: %r' % phase)

    def is_phase_ready(self, phase, num_action_steps):
        """Check if the current phase is ready.

        Args:
            phase: A string variable.
            num_action_steps: Number of steps in the `start` phase.

        Returns:
            The boolean value indicating if the current phase is ready.
        """
        if self.simulator:
            if phase == 'start':
                if num_action_steps >= self.config.SIM.MAX_ACTION_STEPS:
                    logger.debug('The grasping motion is stuck.')
                    return True

            if phase == 'start' or phase == 'end':
                if self.simulator.check_contact(self.robot.arm, self.table):
                    logger.debug('The gripper contacts the table')
                    return True

        if self.robot.is_limb_ready() and self.robot.is_gripper_ready():
            return True
        else:
            return False

    def plot_pose(self,
                  pose,
                  axis_length=1.0,
                  text=None,
                  text_size=1.0,
                  text_color=[0, 0, 0]):
        """Plot a pose or a frame in the debugging visualizer."""
        if not isinstance(pose, Pose):
            pose = Pose(pose)

        origin = pose.position
        x_end = origin + np.dot([axis_length, 0, 0], pose.matrix3.T)
        y_end = origin + np.dot([0, axis_length, 0], pose.matrix3.T)
        z_end = origin + np.dot([0, 0, axis_length], pose.matrix3.T)

        pybullet.addUserDebugLine(
            origin,
            x_end,
            lineColorRGB=[1, 0, 0],
            lineWidth=2)

        pybullet.addUserDebugLine(
            origin,
            y_end,
            lineColorRGB=[0, 1, 0],
            lineWidth=2)

        pybullet.addUserDebugLine(
            origin,
            z_end,
            lineColorRGB=[0, 0, 1],
            lineWidth=2)

        if text is not None:
            pybullet.addUserDebugText(
                text,
                origin,
                text_color,
                text_size)
