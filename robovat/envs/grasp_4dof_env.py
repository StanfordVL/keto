"""Top-down 4-DoF grasping environment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import random

import gym
import numpy as np

from robovat.envs import arm_env
from robovat.envs.observations.camera_obs import CameraObs
from robovat.envs.observations.camera_obs import CameraIntrinsicsObs
from robovat.envs.reward_fns.grasp_reward import GraspReward
from robovat.grasp import Grasp2D
from robovat.math import Pose
from robovat.math import get_transform
from robovat.perception.camera import Kinect2
from robovat.robots import sawyer
from robovat.simulation.camera import BulletCamera
from robovat.utils.logging import logger


GRASPABLE_NAME = 'graspable'


class Grasp4DofEnv(arm_env.ArmEnv):
    """Top-down 4-DoF grasping environment."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=True):
        """Initialize."""
        self.simulator = simulator
        self.config = config or self.default_config
        self.debug = debug

        self.build_camera(
                height=self.config.KINECT2.DEPTH.HEIGHT,
                width=self.config.KINECT2.DEPTH.WIDTH,
                intrinsics=self.config.KINECT2.DEPTH.INTRINSICS,
                translation=self.config.KINECT2.DEPTH.TRANSLATION,
                rotation=self.config.KINECT2.DEPTH.ROTATION,
                intrinsics_noise=self.config.KINECT2.DEPTH.INTRINSICS_NOISE,
                translation_noise=self.config.KINECT2.DEPTH.TRANSLATION_NOISE,
                rotation_noise=self.config.KINECT2.DEPTH.ROTATION_NOISE,
                calibration_path=None)

        # TODO(kuanfang): Add camera parameters observation.
        observations = [
            CameraObs(
                name='depth',
                camera=self.camera,
                modality='depth',
                max_visible_distance_m=None),
            CameraIntrinsicsObs(
                name='intrinsics',
                camera=self.camera)
        ]

        reward_fns = [
            GraspReward(
                name='grasp_reward',
                end_effector_name=sawyer.SawyerSim.ARM_NAME,
                graspable_name=GRASPABLE_NAME)
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

        super(Grasp4DofEnv, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=self.simulator,
            config=self.config,
            debug=self.debug)

        # TODO(kuanfang): Correct this action space.
        if self.config.ACTION.TYPE == '4DOF_CENTER':
            self.action_space = gym.spaces.Box(
                low=np.array([0.3, -0.3, 0.03, 0]),
                high=np.array([0.6, 0.3, 0.1, 2 * np.pi]),
                dtype=np.float32)
        elif self.config.ACTION.TYPE == '4DOF_ENDPOINTS':
            self.action_space = gym.spaces.Box(
                -(2**16 - 1), 2**16 - 1, (5,), dtype=np.float32)
        else:
            raise ValueError

    def build_camera(self,
                     height,
                     width,
                     intrinsics=None,
                     translation=None,
                     rotation=None,
                     intrinsics_noise=None,
                     translation_noise=None,
                     rotation_noise=None,
                     calibration_path=None):
        """Build the camera sensor."""
        if self.simulator:
            self.camera = BulletCamera(simulator=self.simulator,
                                       height=height,
                                       width=width)
        else:
            self.camera = Kinect2(height=height,
                                  width=width)

        if calibration_path:
            intrinsics, translation, rotation = self.camera.load_calibration(
                calibration_path)

        assert intrinsics is not None
        assert translation is not None
        assert rotation is not None

        self.intrinsics = intrinsics
        self.translation = translation
        self.rotation = rotation
        self.intrinsics_noise = intrinsics_noise
        self.translation_noise = translation_noise
        self.rotation_noise = rotation_noise

        self.camera.set_calibration(
            intrinsics, translation, rotation)

    def reset_scene(self):
        """Reset the scene in simulation or the real world.
        """
        super(Grasp4DofEnv, self).reset_scene()

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
        super(Grasp4DofEnv, self).reset_robot()
        self.robot.reset(self.config.ARM.OFFSTAGE_POSITIONS)

    def execute_action(self, action):
        """Execute the grasp action.

        Args:
            action: A 4-dimentional vector represents [x, y, z, angle].

        Returns:
            grasp_success: True or False.
        """
        if self.config.ACTION.TYPE == '4DOF_CENTER':
            x, y, z, angle = action
        elif self.config.ACTION.TYPE == '4DOF_ENDPOINTS':
            grasp = Grasp2D.from_vector(action, camera=self.camera)
            x, y, z, angle = grasp.as_4dof()
        else:
            raise ValueError

        pregrasp_pose = Pose(
            [[x, y, self.config.ARM.GRIPPER_SAFE_HEIGHT], [0, np.pi, angle]])
        grasp_pose = Pose(
            [[x, y, z + self.config.ARM.FINGER_TIP_OFFSET], [0, np.pi, angle]])
        self.grasp_procedure(pregrasp_pose, grasp_pose)

        if self.simulator:
            self.simulator.wait_until_stable(self.graspable, 0.1, None)

    def grasp_procedure(self, pregrasp_pose, grasp_pose):
        """Handle the grasp action.
        """
        status = 'initial'

        while(status != 'done'):
            old_status = status

            if status == 'initial':
                if self.robot.is_limb_ready():
                    status = 'overhead'
                    self.robot.move_to_joint_positions(
                        self.config.ARM.OVERHEAD_POSITIONS)

            elif status == 'overhead':
                if self.robot.is_limb_ready():
                    status = 'pregrasp'
                    self.robot.move_to_gripper_pose(pregrasp_pose)

            elif status == 'pregrasp':
                if self.robot.is_limb_ready():
                    status = 'grasp'
                    self.robot.move_to_gripper_pose(grasp_pose,
                                                    straight_line=True)

                    if self.simulator:
                        # Resolve the stuck grasping motion.
                        num_grasp_steps = 0

                        # Modify frictions for robust simulation.
                        self.robot.l_finger_tip.set_dynamics(
                            lateral_friction=0.001,
                            spinning_friction=0.001)
                        self.robot.r_finger_tip.set_dynamics(
                            lateral_friction=0.001,
                            spinning_friction=0.001)
                        self.table.set_dynamics(
                            lateral_friction=100)

            elif status == 'grasp':
                is_stuck = False
                if self.simulator:
                    num_grasp_steps += 1
                    if num_grasp_steps >= self.config.SIM.MAX_GRASP_STEPS:
                        is_stuck = True
                        logger.debug('The grasping motion is stuck.')

                contact_table = self.simulator.check_contact(
                        self.robot.arm, self.table)

                if contact_table:
                    logger.debug('The gripper contacts the table')

                if self.robot.is_limb_ready() or contact_table or is_stuck:
                    status = 'close_gripper'
                    self.robot.grip(1)

            elif status == 'close_gripper':
                if self.robot.is_gripper_ready():
                    status = 'postgrasp'
                    if self.config.ARM.MOVE_TO_OVERHEAD_AFTER_GRASP:
                        self.robot.move_to_joint_positions(
                            self.config.ARM.OVERHEAD_POSITIONS)
                    else:
                        self.robot.move_to_gripper_pose(pregrasp_pose)

                    if self.simulator:
                        # Modify frictions for robust simulation.
                        self.robot.l_finger_tip.set_dynamics(
                            lateral_friction=100,
                            rolling_friction=10,
                            spinning_friction=10)
                        self.robot.r_finger_tip.set_dynamics(
                            lateral_friction=100,
                            rolling_friction=10,
                            spinning_friction=10)
                        self.table.set_dynamics(
                            lateral_friction=1)

            elif status == 'postgrasp':
                if self.robot.is_limb_ready():
                    status = 'done'

            else:
                raise ValueError('Unrecognized control status: %s' % status)

            if status != old_status:
                logger.debug('Status: %s', status)

            if self.simulator:
                self.simulator.step()
