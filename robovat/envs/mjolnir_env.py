"""The environment of flying hammer-shape tool.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from robovat.envs import robot_env
from robovat.envs.observations import pose_obs
from robovat.robots import mjolnir
from robovat.math import Pose  # NOQA
from robovat.utils.logging import logger


class MjolnirEnv(robot_env.RobotEnv):
    """The environment of flying hammer-shape tool."""

    def __init__(self,
                 simulator=None,
                 config=None,
                 debug=False):
        """Initialize."""
        self.robot = None
        self.ground = None

        observations = [
            pose_obs.PoseObs('mjolnir_body', 'pose2d', name='robot'),
            pose_obs.PoseObs('concave', 'pose2d', name='concave'),
            pose_obs.PoseObs('convex', 'pose2d', name='convex'),
        ]

        reward_fns = []

        super(MjolnirEnv, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=simulator,
            config=config,
            debug=debug)

        self.num_modes = self.config.ACTION.NUM_MODES
        self.num_waypoints = self.config.ACTION.NUM_WAYPOINTS
        self.linear_velocity = (float(self.config.ACTION.POSITION_STEP) /
                                float(self.config.ACTION.WAYPOINT_TIME))
        self.angular_velocity = (float(self.config.ACTION.EULER_STEP) /
                                 float(self.config.ACTION.WAYPOINT_TIME))
        self.waypoint_time = float(self.config.ACTION.WAYPOINT_TIME)

        action_shape = [self.num_waypoints * 3]
        self.action_space = gym.spaces.Dict({
            'action': gym.spaces.Box(
                low=-np.ones(action_shape, dtype=np.float32),
                high=np.ones(action_shape, dtype=np.float32),
                dtype=np.float32),
            'use_mode': gym.spaces.Discrete(2),
            'mode': gym.spaces.Discrete(self.num_modes),
        })

        self.cspace = gym.spaces.Box(
            low=np.array(self.config.CSPACE.LOW),
            high=np.array(self.config.CSPACE.HIGH),
            dtype=np.float32)

        self.movable_bodies = []

    def reset_scene(self):
        """Reset the scene in simulation or the real world."""
        self.movable_bodies = []
        self.ground = self.simulator.add_body(self.config.GROUND.PATH,
                                              self.config.GROUND.POSE,
                                              is_static=True,
                                              name='ground')
        self.wall = self.simulator.add_body(self.config.WALL.PATH,
                                            self.config.WALL.POSE,
                                            is_static=True,
                                            name='wall')
        self.concave = self.add_movable('concave', self.config.CONCAVE)
        self.convex = self.add_movable('convex', self.config.CONVEX)

    def reset_robot(self):
        """Reset the robot in simulation or the real world."""
        body_config = self.config.MJOLNIR

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
            for other_body in self.movable_bodies:
                dist = np.linalg.norm(
                    pose.position[:2] - other_body.position[:2])
                if dist < body_config.MARGIN:
                    is_valid = False
                    break

        self.robot = mjolnir.Mjolnir(simulator=self.simulator, pose=pose)

        while not self.robot.is_ready():
            self.simulator.step()

    def compute_waypoints(self, action):
        waypoints = []

        pose = self.robot.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        z = float(pose.position.z)
        theta = float(pose.euler[2])

        for i in range(self.num_waypoints):
            delta_x = action[i, 0] * self.config.ACTION.POSITION_STEP
            delta_y = action[i, 1] * self.config.ACTION.POSITION_STEP
            delta_theta = action[i, 2] * self.config.ACTION.EULER_STEP

            x = x + delta_x * np.cos(theta) - delta_y * np.sin(theta)
            y = y + delta_x * np.sin(theta) + delta_y * np.cos(theta)
            theta = (theta + delta_theta) % (2 * np.pi)

            x = np.clip(x, self.cspace.low[0], self.cspace.high[0])
            y = np.clip(y, self.cspace.low[1], self.cspace.high[1])
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            waypoint = Pose([[x, y, z], [pose.euler[0], pose.euler[1], theta]])
            waypoints.append(waypoint)

        return waypoints

    def execute_action(self, action):
        """Execute the robot action.
        """
        action = action['action']
        action = np.reshape(action, [self.num_waypoints, 3])

        if self.debug:
            logger.info('Step: %d; Executing action %r', self.num_steps, action)

        waypoints = self.compute_waypoints(action)

        if self.debug:
            print('Waypoints:')
            for i in range(self.num_waypoints):
                waypoint = waypoints[i]
                print('%s' % (waypoint))
            self.visualize_waypoints(waypoints)

        waypoint_id = 0
        while True:
            self.simulator.step()

            if self.robot.is_ready():
                input('Input: ')

                if waypoint_id >= self.num_waypoints:
                    break

                waypoint = waypoints[waypoint_id]

                if self.debug:
                    logger.info('Moving to waypoint %d, from %s to %s',
                                waypoint_id, self.robot.pose, waypoint)

                self.robot.move_to(
                    waypoint,
                    linear_velocity=self.linear_velocity,
                    angular_velocity=self.angular_velocity,
                    timeout=self.waypoint_time)
                waypoint_id += 1

    def visualize_waypoints(self, waypoints):
        self.simulator.clear_visualization()
        for i in range(self.num_waypoints):
            waypoint = waypoints[i]
            self.simulator.plot_pose(waypoint, 0.05, '%d' % (i))

        for i in range(self.num_waypoints):
            if i == 0:
                self.simulator.plot_line(self.robot.pose.position,
                                         waypoints[0].position)
            else:
                self.simulator.plot_line(waypoints[i - 1].position,
                                         waypoints[i].position)

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
            for other_body in self.movable_bodies:
                dist = np.linalg.norm(
                    pose.position[:2] - other_body.position[:2])
                if dist < body_config.MARGIN:
                    is_valid = False
                    break

        # Add the body.
        path = body_config.PATH
        body = self.simulator.add_body(path, pose, name=body_name)
        self.movable_bodies.append(body)

        return body
