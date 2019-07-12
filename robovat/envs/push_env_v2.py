"""Multi-stage task environment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import random

import gym
from matplotlib import pyplot as plt
import numpy as np

from robovat.envs import robot_env
from robovat.envs import arm_env
from robovat.envs.observations import attribute_obs
from robovat.envs.observations import camera_obs
from robovat.envs.observations import pose_obs
from robovat.envs.reward_fns import manip_reward
from robovat.math import Pose
from robovat.utils.logging import logger


SUCCESS_THRESH = 500.


class PushEnvV2(arm_env.ArmEnv):
    """Multi-stage task environment."""

    def __init__(self,
                 task_name=None,
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
                calibration_path=None,
                crop=self.config.KINECT2.DEPTH.CROP)

        self.task_config = self.config.TASK[task_name.upper()]
        self.movable_configs = [
            self.task_config[config_name]
            for config_name in self.task_config.CONFIG_NAMES]

        if self.simulator:
            # Movable bodies.
            name_to_movable_paths = {}
            for config_name in self.task_config.CONFIG_NAMES:
                movable_config = self.task_config[config_name]
                movable_paths = []
                for pattern in movable_config.PATHS:
                    movable_paths += glob.glob(pattern)
                name_to_movable_paths[config_name] = movable_paths

            self.movable_paths = [
                name_to_movable_paths[config_name]
                for config_name in self.task_config.CONFIG_NAMES]

            self.movable_names = ['movable_%d' % i for i in
                                  range(len(self.movable_configs))]

            self.movable_bodies = []
            self.movable_margin = self.task_config.MARGIN 

            # Observations.
            observations = [
                camera_obs.SegmentedPointCloudObs(
                    self.camera, self.config.OBS.NUM_POINTS, self.movable_names,
                    name='point_cloud'),
                pose_obs.PoseObs(
                    self.movable_names, 'position', name='position'),
                attribute_obs.FlagObs('is_safe', name='is_safe'),
                attribute_obs.FlagObs('is_effective', name='is_effective'),
                attribute_obs.RandomIndexObs(name='episode_id'),
            ]

        else:
            raise NotImplementedError
            # TODO(kuanfang): Implement point cloud segmentation in real world.

        if task_name is None:
            reward_fns = [
                manip_reward.ManipReward(
                    name='toy_reward',
                    task_name=config.TASK_NAME)
            ]
            self.tile_config = None
        elif task_name == 'data_collection':
            reward_fns = [
                manip_reward.ManipReward(
                    name='reward',
                    task_name=None)
            ]
            self.tile_config = None
        elif task_name == 'gather':
            reward_fns = [
                manip_reward.ManipReward(
                    name='reward',
                    task_name='gather')
            ]
            self.tile_config = {
                'clearance': manip_reward.GATHER_TILES,
            }
        elif task_name == 'insert':
            reward_fns = [
                manip_reward.ManipReward(
                    name='reward',
                    task_name='insert')
            ]
            self.tile_config = {
                'slot': manip_reward.SLOT_TILES,
                'goal': manip_reward.INSERT_GOAL_TILES,
            }
        elif task_name == 'cross':
            reward_fns = [
                manip_reward.ManipReward(
                    name='reward',
                    task_name='wide_cross')  # TODO
            ]
            self.tile_config = {
                'path': manip_reward.PATH_TILES,
                'goal': manip_reward.CROSS_GOAL_TILES,
            }
        else:
            raise ValueError('Unrecognized task name: %r' % task_name)

        super(PushEnvV2, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=self.simulator,
            config=self.config,
            debug=self.debug)

        # Action space.
        self.num_waypoints = self.config.ACTION.NUM_WAYPOINTS
        self.action_space = gym.spaces.Dict({
            'start': gym.spaces.Box(
                low=-np.ones([2], dtype=np.float32),
                high=np.ones([2], dtype=np.float32),
                dtype=np.float32),
            'motion': gym.spaces.Box(
                low=-np.ones([self.num_waypoints * 2], dtype=np.float32),
                high=np.ones([self.num_waypoints * 2], dtype=np.float32),
                dtype=np.float32),
        })

        # Configuration space.
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
        self.attributes = {
            'is_safe': None,
            'is_effective': None,
        }
        self.start_status = None
        self.end_status = None
        self.num_action_steps = None

        # Statistics.
        self.num_total_steps = 0
        self.num_unsafe = 0
        self.num_ineffective = 0
        self.num_useful = 0
        self.num_successes = 0

        self.num_successes_by_step = [0] * int(self.config.MAX_STEPS + 1)

        # Visualization.
        if self.debug:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            self.axs = axs

            plt.ion()
            plt.show()

    def step(self, action):
        observation, reward, is_done, info = super(PushEnvV2, self).step(action)

        if is_done and reward >= SUCCESS_THRESH:
            self.num_successes += 1
            self.num_successes_by_step[self._num_steps] += 1

        if is_done:
            logger.info(
                'num_successes: %d, success_rate: %.3f',
                self.num_successes,
                self.num_successes / float(self._num_episodes + 1e-14),
            )
            text = ('num_successes_by_step: ' +
                    ', '.join(['%d'] * int(self.config.MAX_STEPS + 1)))

            logger.info(text, *self.num_successes_by_step)

            logger.info(
                'num_total_steps %d, '
                'unsafe: %.3f, ineffective: %.3f, useful: %.3f',
                self.num_total_steps,
                self.num_unsafe / float(self.num_total_steps + 1e-14),
                self.num_ineffective / float(self.num_total_steps + 1e-14),
                self.num_useful / float(self.num_total_steps + 1e-14))

        return observation, reward, is_done, info

    def reset_scene(self):
        """Reset the scene in simulation or the real world.
        """
        super(PushEnvV2, self).reset_scene()

        # Simulation.
        if self.simulator:
            self.num_action_steps = 0

            path = self.config.SIM.FENCE.PATH
            self.fence = self.simulator.add_body(
                path, self.table.pose, is_static=True, name='fence')

            self.load_tiles()
            self.load_bodies()

        # Attributes
        self.attributes = {
            'is_safe': True,
            'is_effective': True,
        }

    def load_tiles(self):
        if self.tile_config is not None:
            for tile_config in self.tile_config.values():
                path = tile_config.PATH
                center = tile_config.POSE
                size = tile_config.SIZE
                rgba = tile_config.RGBA
                specular = tile_config.SPECULAR
                for i, offset in enumerate(tile_config.OFFSETS):
                    pose = np.array(center) + np.array(offset) * size
                    name = 'tile_%d' % i
                    self.add_tile(path, pose, rgba, specular, name)

    def load_bodies(self):
        is_valid = False
        while not is_valid:
            logger.info('Loading movable objects...')
            is_valid = True
            self.movable_bodies = []
            self.static_bodies = []

            for i, name in enumerate(self.movable_names):
                body = self.add_movable_body(
                    name,
                    self.movable_paths[i],
                    self.movable_configs[i],
                    max_attemps=32)
                if body is False:
                    is_valid = False
                    break
                else:
                    self.movable_bodies.append(body)

            if is_valid:
                for i, name in enumerate(self.movable_names):
                    body = self.simulator.bodies[name]
                    if body.position.z < self.table_pose.position.z:
                        is_valid = False
                        break

            if not is_valid:
                logger.info('Invalid arrangement, reset the scene...')
                for i, body in enumerate(self.movable_bodies):
                    self.simulator.remove_body(body.name)

        logger.info('Waiting for movable objects to be stable...')
        self.simulator.wait_until_stable(self.movable_bodies)

    def add_movable_body(self, name, body_paths, body_config, max_attemps):
        """Add an movable object."""
        # Sample a pose which is not overlapped with other objects.
        num_attemps = 0
        is_valid = False
        while not is_valid and num_attemps <= max_attemps:
            pose = Pose.uniform(x=body_config.POSE.X,
                                y=body_config.POSE.Y,
                                z=body_config.POSE.Z,
                                roll=body_config.POSE.ROLL,
                                pitch=body_config.POSE.PITCH,
                                yaw=body_config.POSE.YAW)
            is_valid = True
            for other_body in self.movable_bodies:
                dist = np.linalg.norm(
                    pose.position[:2] - other_body.position[:2])
                if dist < self.movable_margin:
                    is_valid = False
                    num_attemps += 1
                    break

        if not is_valid:
            return False

        # Add the body.
        path = random.choice(body_paths)
        scale = np.random.uniform(*body_config.SCALE)
        body = self.simulator.add_body(path, pose, scale, name=name)

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

    def add_tile(self, path, pose, rgba, specular, name=None):
        pose = Pose.uniform(x=pose[0], y=pose[1], z=pose[2])
        body = self.simulator.add_body(
            path, pose, is_static=True, name=name)
        body.set_color(rgba=rgba, specular=specular)
        return body

    def execute_action(self, action):
        """Execute the grasp action.

        Args:
            action: A dictionary of mode and argument of the action.
        """

        # TODO: Debug
        return

        self.attributes = {
            'is_safe': True,
            'is_effective': True,
        }

        waypoints = self.compute_waypoints(action)
        if self.debug:
            logger.debug('step: %d, start: (%.2f, %.2f), motion: %r',
                         self.num_steps,
                         action['start'][0],
                         action['start'][1],
                         list(action['motion']))

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
                        waypoints[0].z) <= 0.01:
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

    def compute_waypoints(self, action):
        """Execute the robot action.
        """
        start = action['start']
        motion = np.reshape(action['motion'], [self.num_waypoints, 2])

        x = start[0] * self.start_range[0] + self.start_offset[0]
        y = start[1] * self.start_range[1] + self.start_offset[1]
        z = self.config.ARM.FINGER_TIP_OFFSET + self.start_offset[2]
        angle = 0.0

        start = Pose(
            [[x, y, z], [np.pi, 0, (angle + np.pi) % (2 * np.pi) - np.pi]])
        waypoints = [start]

        for i in range(self.num_waypoints):
            delta_x = motion[i, 0] * self.config.ACTION.MOTION.TRANSLATION_X
            delta_y = motion[i, 1] * self.config.ACTION.MOTION.TRANSLATION_Y

            x = x + delta_x
            y = y + delta_y

            x = np.clip(x, self.cspace.low[0], self.cspace.high[0])
            y = np.clip(y, self.cspace.low[1], self.cspace.high[1])

            waypoint = Pose(
                [[x, y, z], [np.pi, 0, (angle + np.pi) % (2 * np.pi) - np.pi]])
            waypoints.append(waypoint)

        return waypoints

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

                # TODO(Debug)
                # if self.debug:
                #     logger.debug('phase: %s', next_phase)

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
            if self.simulator.check_contact(
                    self.robot.arm, self.movable_bodies):
                logger.warning('Unsafe action: Movable bodies stuck on robots.')
                return False

            if self.simulator.check_contact(
                    [self.ground, self.robot.base], self.movable_bodies):
                logger.warning('Unsafe action: Movable bodies drop to ground.')
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
        if self.simulator:
            positions = [body.position for body in self.movable_bodies]
            angles = [body.euler[2] for body in self.movable_bodies]
            return [np.stack(positions, axis=0), np.stack(angles, axis=0)]
        
        return None

    def plot_waypoints(self, ax, waypoints, c='lawngreen', alpha=1.0):
        p1 = None
        p2 = None
        for i, waypoint in enumerate(waypoints):
            p1 = self.camera.project_point(waypoint.position)
            if i == 0:
                ax.scatter(p1[0], p1[1],
                           c=c, alpha=alpha, s=2.0)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        c=c, alpha=alpha, linewidth=1.0)
            p2 = p1

    def plot_pred_state(self, ax, state, pred_state, termination):
        z = self.config.ARM.FINGER_TIP_OFFSET
        num_bodies = len(self.movable_configs)
        num_steps = pred_state.shape[0]
        for i in range(num_bodies):
            p1 = self.camera.project_point(list(state[i]) + [z])
            if i == 0:
                c = 'orange'
            elif i == 1:
                c = 'lawngreen'
            elif i == 2:
                c = 'dodgerblue'
            else:
                c = 'red'

            for t in range(num_steps):
                p2 = self.camera.project_point(list(pred_state[t, i]) + [z])

                if np.linalg.norm(p2 - p1) >= 0.1:
                    ax.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1],
                             head_width=15, head_length=10,
                             fc=c, ec=c,
                             zorder=100)
                    p1 = p2

                # TODO
                if termination[t]:
                    break

    def plot_waypoints_in_simulation(self, waypoints):
        self.simulator.clear_visualization()
        for i in range(len(waypoints)):
            waypoint = waypoints[i]
            self.simulator.plot_pose(waypoint, 0.05, '%d' % (i))

        for i in range(1, len(waypoints)):
            self.simulator.plot_line(waypoints[i - 1].position,
                                     waypoints[i].position)

    def visualize(self, action, info):
        if not self.debug:
            return

        # Reset.
        images = self.camera.frames()
        rgb = images['rgb']
        for ax in self.axs.flat:
            ax.cla()
            ax.imshow(rgb)

        # Plot sampled actions.
        if 'start' in info:
            print('action')

            num_samples = info['start'].shape[0]
            # max_plots = min(num_samples, self.config.ACTION.MAX_PLOTS)
            # num_plots = [0] * 4

            i = 0 
            while i < num_samples - 1:
                i += 1
                mode_i = info['c_id'][i]
                valid_i = info['valid'][i]

                print(i, mode_i, valid_i)

                # if not valid_i:
                #     continue
                #
                # if num_plots[mode_i] >= max_plots:
                #     continue
                #
                # num_plots[mode_i] += 1
                #
                # action_i = {
                #     'start': info['start'][i],
                #     'motion': info['motion'][i],
                # }
                # waypoints_i = self.compute_waypoints(action_i)
                # if valid_i:
                #     c = 'lawngreen'
                # else:
                #     c = 'r'
                # ax = self.axs.flat[mode_i]
                # self.plot_waypoints(ax, waypoints_i, c=c, alpha=0.8)

        # Plot predicted states.
        if 'pred_state' in info:
            num_modes = info['pred_state'].shape[1]

            print('pred_state')
            for mode_i in range(num_modes):
                print(mode_i, num_modes)

                ax = self.axs.flat[mode_i]
                state = info['state']
                pred_state = info['pred_state'][:, mode_i]
                termination_i = info['termination'][:, mode_i]
                self.plot_pred_state(ax, state, pred_state, termination_i)

        # Plot waypoints in simulation.
        # if self.simulator:
        #     waypoints = self.compute_waypoints(action)
        #     self.plot_waypoints_in_simulation(waypoints)

        plt.draw()
        plt.pause(1e-3)

        # while(1):
        #     self.simulator.step()

        input('Enter')


class DataCollectionEnv(PushEnvV2):
    """Gathering task environment."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=True):
        super(DataCollectionEnv, self).__init__(
            task_name='data_collection',
            simulator=simulator,
            config=config,
            debug=debug)


class GatherEnv(PushEnvV2):
    """Gathering task environment."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=True):
        super(GatherEnv, self).__init__(
            task_name='gather',
            simulator=simulator,
            config=config,
            debug=debug)


class InsertEnv(PushEnvV2):
    """Inserting task environment."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=True):
        super(InsertEnv, self).__init__(
            task_name='insert',
            simulator=simulator,
            config=config,
            debug=debug)


class CrossEnv(PushEnvV2):
    """Crossing task environment."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=True):
        super(CrossEnv, self).__init__(
            task_name='cross',
            simulator=simulator,
            config=config,
            debug=debug)
