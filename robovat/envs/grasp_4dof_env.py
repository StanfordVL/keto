
import glob
import random

import gym
import numpy as np

from robovat.envs import arm_env
from robovat.envs.observations import camera_obs
from robovat.envs.reward_fns.grasp_reward import GraspReward
from robovat.grasp import Grasp2D
from robovat.math import Pose
from robovat.math import get_transform
from robovat.robots import sawyer
from robovat.utils.logging import logger


GRASPABLE_NAME = 'graspable'


class Grasp4DofEnv(arm_env.ArmEnv):
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
            camera_obs.CameraObs(
                name=self.config.OBSERVATION.TYPE,
                camera=self.camera,
                modality=self.config.OBSERVATION.TYPE,
                max_visible_distance_m=None),
            camera_obs.SegmentedPointCloudObs(
                camera=self.camera,
                num_points=self.config.OBSERVATION.NUM_POINTS,
                body_names=['graspable'],
                name='point_cloud'),
            camera_obs.DeprojectParamsObs(
                camera=self.camera,
                num_points=self.config.OBSERVATION.NUM_POINTS,
                name='deproject_params'),
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

            self.init_record()

        super(Grasp4DofEnv, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=self.simulator,
            config=self.config,
            debug=self.debug)

        if self.config.ACTION.TYPE == 'CUBOID':
            low = self.config.ACTION.CUBOID.LOW + [0.0]
            high = self.config.ACTION.CUBOID.HIGH + [2 * np.pi]
            self.action_space = gym.spaces.Box(
                low=np.array(low),
                high=np.array(high),
                dtype=np.float32)
        elif self.config.ACTION.TYPE == 'IMAGE':
            height = self.camera.height
            width = self.camera.width
            self.action_space = gym.spaces.Box(
                low=np.array([0, 0, 0, 0, -(2*24 - 1)]),
                high=np.array([width, height, width, height, 2*24 - 1]),
                dtype=np.float32)
        else:
            raise ValueError

    def init_record(self):
        self.success_record = np.zeros(
             shape=(len(self.all_graspable_paths), ))
        return

    def feedback(self, reward):
        self.success_record[self.graspable_index] += reward
        print(self.success_record)
        return

    def choose_graspable_index(self, explore_prob=0.7):
        if np.random.uniform() < explore_prob:
            return np.random.randint(len(self.all_graspable_paths))
        record = self.success_record + np.random.normal(
                size=np.shape(self.success_record))
        argmin = np.argmin(record)
        return argmin

    def reset_scene(self):
        """Reset the scene in simulation or the real world.
        """
        super(Grasp4DofEnv, self).reset_scene()

        # Reload graspable object.
        if self.config.SIM.GRASPABLE.RESAMPLE_N_EPISODES:
            if (self.num_episodes %
                    self.config.SIM.GRASPABLE.RESAMPLE_N_EPISODES == 0):
                self.graspable_path = None
    
        self.graspable_index = self.choose_graspable_index()
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
        super(Grasp4DofEnv, self).reset_robot()
        self.robot.reset(self.config.ARM.OFFSTAGE_POSITIONS)

    def execute_action(self, action):
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
            [[x, y, z + self.config.ARM.FINGER_TIP_OFFSET], [0, np.pi, angle]])

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
                logger.debug('phase: %s', phase)

                if phase == 'overhead':
                    self.robot.move_to_joint_positions(
                        self.config.ARM.OVERHEAD_POSITIONS)
                    # self.robot.grip(0)

                elif phase == 'prestart':
                    prestart = start.copy()
                    prestart.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(prestart)

                elif phase == 'start':
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

                elif phase == 'postend':
                    postend = self.robot.end_effector.pose
                    postend.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(postend, straight_line=True)

                    # Prevent problems caused by unrealistic frictions.
                    if self.simulator:
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
