"""Multi-stage task environment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import random

import gym
import numpy as np

from robovat.envs import robot_env
from robovat.envs import arm_env
from robovat.envs.observations import attribute_obs
from robovat.envs.observations import camera_obs
from robovat.envs.observations import pose_obs
from robovat.envs.reward_fns import manip_reward
from robovat.math import Pose
from robovat.utils.logging import logger


def convert_state(observation):
    pose = observation['pose']
    position = pose[:, :2]
    euler = pose[0, 5]
    euler = np.stack([np.cos(euler), np.sin(euler)], axis=-1)
    state = {
        'position': position,
        'euler': euler,
    }
    return state


class ManipEnv(arm_env.ArmEnv):
    """Multi-stage task environment."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=True):
        """Initialize."""
        self.simulator = simulator
        self.config = config or self.default_config
        self.debug = debug

        self.num_modes = self.config.ACTION.NUM_MODES
        self.num_start_angles = self.config.ACTION.NUM_START_ANGLES
        self.num_waypoints = self.config.ACTION.NUM_WAYPOINTS

        if self.simulator:
            movable_name = self.config.SIM.MOVABLE
            if movable_name == 'soda':
                self.movable_config = self.config.SIM.SODA
            elif movable_name == 'convex':
                self.movable_config = self.config.SIM.CONVEX
            elif movable_name == 'ycb':
                self.movable_config = self.config.SIM.YCB
            elif movable_name == 'fixed':
                self.movable_config = self.config.SIM.FIXED
            elif movable_name == 'toy':
                self.movable_config = self.config.SIM.TOY
            else:
                raise ValueError('Unrecognized movable object set: %r.'
                                 % (movable_name))

            self.movable_names = []
            for i in range(self.config.OBS.NUM_MOVABLES):
                self.movable_names.append('movable_%d' % i)

            self.movable_paths = []
            for pattern in self.movable_config.PATHS:
                self.movable_paths += glob.glob(pattern)

            self.movables = []

        else:
            raise NotImplementedError

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
                self.camera, self.config.OBS.NUM_POINTS, self.movable_names,
                name='point_cloud'),
            pose_obs.PoseObs(self.movable_names, 'pose', name='pose'),
            attribute_obs.FlagObs('is_safe', name='is_safe'),
            attribute_obs.FlagObs('is_effective', name='is_effective'),
        ]

        # TODO(kuanfang0
        reward_fns = [
            manip_reward.ManipReward(
                name='toy_reward',
                task_name=config.TASK_NAME)
        ]

        super(ManipEnv, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=self.simulator,
            config=self.config,
            debug=self.debug)

        motion_shape = [self.num_waypoints * 3]
        self.action_space = gym.spaces.Dict({
            'start_position': gym.spaces.Box(
                low=-np.ones([3], dtype=np.float32),
                high=np.ones([3], dtype=np.float32),
                dtype=np.float32),
            'start_angle': gym.spaces.Discrete(self.num_start_angles),
            'motion': gym.spaces.Box(
                low=-np.ones(motion_shape, dtype=np.float32),
                high=np.ones(motion_shape, dtype=np.float32),
                dtype=np.float32),
            'use_mode': gym.spaces.Discrete(2),
            'mode': gym.spaces.Discrete(self.num_modes + 1),
        })

        self.cspace = gym.spaces.Box(
            low=np.array(self.config.ACTION.CSPACE.LOW),
            high=np.array(self.config.ACTION.CSPACE.HIGH),
            dtype=np.float32)

        start_low = np.array(self.config.ACTION.CSPACE.LOW, dtype=np.float32)
        start_high = np.array(self.config.ACTION.CSPACE.HIGH, dtype=np.float32)
        self.start_offset = 0.5 * (start_high + start_low)
        self.start_range = 0.5 * (start_high - start_low)

        # Execution phases.
        self.phase_list = ['initial',
                           'overhead',
                           'pre',
                           'start',
                           'motion',
                           'post',
                           'offstage',
                           'done']

        # Action related information.
        self.num_action_steps = None
        self.start_status = None
        self.end_status = None
        self.attributes = {
            'is_safe': None,
            'is_effective': None,
        }

        # Statistics.
        self.num_total_steps = 0
        self.num_unsafe = 0
        self.num_ineffective = 0
        self.num_useful = 0

    def reset_scene(self):
        """Reset the scene in simulation or the real world.
        """
        logger.info(
            'num_total_steps %d, '
            'unsafe: %.3f, ineffective: %.3f, useful: %.3f',
            self.num_total_steps,
            float(self.num_unsafe) / (self.num_total_steps + 1e-14),
            float(self.num_ineffective) / (self.num_total_steps + 1e-14),
            float(self.num_useful) / (self.num_total_steps + 1e-14))

        super(ManipEnv, self).reset_scene()
        path = self.config.SIM.FENCE.PATH
        self.fence = self.simulator.add_body(
            path, self.table.pose, is_static=True, name='fence')

        # Load movable objects.
        is_valid = False
        while not is_valid:
            logger.info('Loading movable objects...')
            self.movables = []
            for i, body_name in enumerate(self.movable_names):
                body = self.add_movable(body_name, self.movable_config)
                self.movables.append(body)

            is_valid = True
            for i, body_name in enumerate(self.movable_names):
                body = self.simulator.bodies[body_name]
                if body.position.z < self.table_pose.position.z:
                    is_valid = False
                    break

            if not is_valid:
                logger.info('Invalid arrangement, reset the scene...')
                for i, body_name in enumerate(self.movable_names):
                    self.simulator.remove_body(body_name)

        logger.info('Waiting for movable objects to be stable...')
        self.simulator.wait_until_stable(self.movables)

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

        # Simulation.
        if self.simulator:
            self.num_action_steps = 0
        else:
            raise NotImplementedError

        # Attributes
        self.attributes = {
            'is_safe': True,
            'is_effective': True,
        }

    def reset_robot(self):
        """Reset the robot in simulation or the real world."""
        super(ManipEnv, self).reset_robot()
        self.robot.grip(1)

    def add_movable(self, body_name, body_config):
        """Add an movable object."""
        # Sample a pose which is not overlapped with other objects.
        is_valid = False
        while not is_valid:
            pose = Pose.uniform(x=body_config.POSE.X,
                                y=body_config.POSE.Y,
                                z=body_config.POSE.Z,
                                roll=body_config.POSE.ROLL,
                                pitch=body_config.POSE.PITCH,
                                yaw=body_config.POSE.YAW)
            is_valid = True
            for other_body in self.movables:
                dist = np.linalg.norm(
                    pose.position[:2] - other_body.position[:2])
                if dist < body_config.MARGIN:
                    is_valid = False
                    break

        # Add the body.
        if body_config.RANDOM_OBJECT:
            path = random.choice(self.movable_paths)
        else:
            body_ind = len(self.movables)
            path = self.movable_paths[body_ind]

        scale = np.random.uniform(*body_config.SCALE)
        body = self.simulator.add_body(path, pose, scale, name=body_name)

        # Wait the object to be dropped onto the table.
        self.simulator.wait_until_stable(
            body,
            linear_velocity_threshold=0.1,
            angular_velocity_threshold=0.1,
            max_steps=500)

        # Change physical properties.
        mass = robot_env.get_property(body_config.MASS)
        lateral_friction = robot_env.get_property(body_config.FRICTION)
        body.set_dynamics(
            mass=mass,
            lateral_friction=lateral_friction,
            rolling_friction=None,
            spinning_friction=None)

        return body

    def compute_waypoints(self, action):
        """Execute the robot action.
        """
        start_position = action['start_position']
        start_angle = action['start_angle']
        motion = np.reshape(action['motion'], [self.num_waypoints, 3]) # NOQA

        x = start_position[0] * self.start_range[0] + self.start_offset[0]
        y = start_position[1] * self.start_range[1] + self.start_offset[1]
        z = start_position[2] * self.start_range[2] + self.start_offset[2]
        z += self.config.ARM.FINGER_TIP_OFFSET
        angle = 2 * np.pi * start_angle / self.num_start_angles

        start = Pose(
            [[x, y, z],
             [np.pi, 0, (angle + np.pi) % (2 * np.pi) - np.pi]])
        waypoints = [start]

        for i in range(self.num_waypoints):
            delta_x = motion[i, 0] * self.config.ACTION.MOTION.TRANSLATION_X
            delta_y = motion[i, 1] * self.config.ACTION.MOTION.TRANSLATION_Y
            delta_angle = motion[i, 2] * self.config.ACTION.MOTION.ROTATION

            # TODO(debug): Poke.
            # delta_x = 1.0 * self.config.ACTION.MOTION.TRANSLATION_X
            # delta_y = 0 * self.config.ACTION.MOTION.TRANSLATION_Y
            # delta_angle = 0

            # TODO(debug): Sweep.
            # delta_x = 0 * self.config.ACTION.MOTION.TRANSLATION_X
            # delta_y = 1.0 * self.config.ACTION.MOTION.TRANSLATION_Y
            # delta_angle = 0

            # TODO(debug): Rotate.
            # delta_x = 0
            # delta_y = 0
            # delta_angle = 1.0 * self.config.ACTION.MOTION.ROTATION

            x = x + delta_x * np.cos(angle) - delta_y * np.sin(angle)
            y = y + delta_x * np.sin(angle) + delta_y * np.cos(angle)
            angle = angle + delta_angle

            x = np.clip(x, self.cspace.low[0], self.cspace.high[0])
            y = np.clip(y, self.cspace.low[1], self.cspace.high[1])

            waypoint = Pose(
                [[x, y, z],
                 [np.pi, 0, (angle + np.pi) % (2 * np.pi) - np.pi]])
            waypoints.append(waypoint)

        return waypoints

    def execute_action(self, action):
        """Execute the grasp action.

        Args:
            action: A dictionary of mode and argument of the action.
        """
        self.attributes = {
            'is_safe': True,
            'is_effective': True,
        }

        logger.info('step: %d, start: (%.2f, %.2f, %.2f, %d), '
                    'use_mode: %d, mode: %d',
                    self.num_steps,
                    action['start_position'][0],
                    action['start_position'][1],
                    action['start_position'][2],
                    action['start_angle'],
                    action['use_mode'],
                    action['mode'],
                    )

        waypoints = self.compute_waypoints(action)

        if self.debug:
            self.visualize_waypoints(waypoints)

        phase = 'initial'
        self.start_status = self.get_movable_status()
        while(phase != 'done'):

            if self.simulator:
                self.simulator.step()
                if phase == 'start' or phase == 'end':
                    self.num_action_steps += 1

            if not self.simulator.num_steps % self.config.SIM.STEPS_CHECK == 0:
                continue

            # Phase transition.
            if self.is_phase_ready(phase):
                phase = self.get_next_phase(phase)

                if phase == 'overhead':
                    self.robot.move_to_joint_positions(
                        self.config.ARM.OVERHEAD_POSITIONS)

                elif phase == 'pre':
                    pose = waypoints[0].copy()
                    pose.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(pose)

                elif phase == 'start':
                    self.robot.move_to_gripper_pose(waypoints[0])

                elif phase == 'motion':
                    self.robot.move_along_gripper_path(waypoints[1:])
                    self.num_action_steps = 0

                elif phase == 'post':
                    pose = self.robot.end_effector.pose
                    pose.z = self.config.ARM.GRIPPER_SAFE_HEIGHT
                    self.robot.move_to_gripper_pose(pose)
        
                if phase == 'offstage':
                    self.robot.move_to_joint_positions(
                        self.config.ARM.OFFSTAGE_POSITIONS)

            # Check interruption.
            interrupt = False

            if phase == 'motion':
                if self.is_singularity():
                    interrupt = True

            if phase == 'overhead' or phase == 'pre' or phase == 'start':
                if phase == 'start' and abs(
                        self.robot.end_effector.position.z -
                        waypoints[0].z) <= 0.03:
                    pass
                else:
                    if not self.is_action_safe():
                        interrupt = True
                        self.attributes['is_safe'] = False

            if interrupt:
                phase = 'offstage'
                self.robot.move_to_joint_positions(
                    self.config.ARM.OFFSTAGE_POSITIONS)

        # Update attributes.
        self.end_status = self.get_movable_status()
        self.attributes['is_effective'] = self.is_action_effective()

        self.num_total_steps += 1
        self.num_unsafe += int(not self.attributes['is_safe'])
        self.num_ineffective += int(not self.attributes['is_effective'])
        self.num_useful += int(self.attributes['is_safe'] and
                               self.attributes['is_effective'])

    def get_next_phase(self, phase):
        """Get the next phase of the current phase.

        Args:
            phase: A string variable.

        Returns:
            The next phase as a string variable.
        """
        if phase in self.phase_list:
            i = self.phase_list.index(phase)
            if i >= len(self.phase_list):
                raise ValueError('phase %r does not have a next phase.')
            else:
                next_phase = self.phase_list[i + 1]

                if self.debug:
                    logger.debug('phase: %s', next_phase)

                return next_phase
        else:
            raise ValueError('Unrecognized phase: %r' % phase)

    def is_phase_ready(self, phase):
        """Check if the current phase is ready.

        Args:
            phase: A string variable.

        Returns:
            The boolean value indicating if the current phase is ready.
        """
        if self.simulator:
            if phase == 'motion':
                if self.num_action_steps >= self.config.SIM.MAX_ACTION_STEPS:
                    if self.config.DEBUG:
                        logger.debug('[phase ready]: The action is stuck.')
                    self.robot.arm.reset_targets()
                    return True

        if self.robot.is_limb_ready() and self.robot.is_gripper_ready():
            self.robot.arm.reset_targets()
            return True

        return False

    def is_action_safe(self):
        """Check if the action is safe."""
        if self.simulator:
            if self.simulator.check_contact(self.robot.arm, self.movables):
                if self.config.DEBUG:
                    logger.warning('Unsafe action.')
                return False

        return True

    def is_action_effective(self):
        """Check if the action is effective."""
        if self.simulator:
            delta_position = np.linalg.norm(
                self.end_status[0] - self.start_status[0], axis=-1)
            delta_position = np.sum(delta_position)

            delta_angle = self.end_status[1] - self.start_status[1]
            delta_angle = (delta_angle + np.pi) % (2 * np.pi) - np.pi
            delta_angle = np.abs(delta_angle)
            delta_angle = np.sum(delta_angle)

            if (delta_position <= self.config.ACTION.MIN_DELTA_POSITION and 
                    delta_angle <= self.config.ACTION.MIN_DELTA_ANGLE):
                if self.config.DEBUG:
                    logger.warning('Ineffective action.')
                return False

        return True

    def is_singularity(self):
        if self.simulator:
            if self.simulator.check_contact(self.robot.arm,
                                            [self.table, self.fence]):
                if self.config.DEBUG:
                    logger.warning('Arm collides with the table.')
                return True

        return False

    def get_movable_status(self):
        positions = [body.position for body in self.movables]
        angles = [body.euler[2] for body in self.movables]
        return [np.stack(positions, axis=0), np.stack(angles, axis=0)]

    def get_state(self):
        obs_data = self.get_observation()
        state = convert_state(obs_data)
        return state

    def visualize_waypoints(self, waypoints):
        self.simulator.clear_visualization()
        for i in range(len(waypoints)):
            waypoint = waypoints[i]
            self.simulator.plot_pose(waypoint, 0.05, '%d' % (i))

            # TODO(debug):
            # print('waypoint [%d]: %s' % (i, waypoint))

        for i in range(1, len(waypoints)):
            self.simulator.plot_line(waypoints[i - 1].position,
                                     waypoints[i].position)

        # TODO(debug):
        # input('input: ')
