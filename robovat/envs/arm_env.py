"""The environment of robot arm.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.envs import robot_env
from robovat.math import Pose
from robovat.perception.camera import Kinect2
from robovat.robots import sawyer
from robovat.simulation.camera import BulletCamera


class ArmEnv(robot_env.RobotEnv):
    """The environment of robot arm."""

    def __init__(self,
                 observations,
                 reward_fns,
                 simulator=None,
                 config=None,
                 debug=False):
        """Initialize."""
        super(ArmEnv, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=simulator,
            config=config,
            debug=debug)

        self.robot = None
        self.ground = None
        self.table = None
        self.table_pose = None

    def reset_scene(self):
        """Reset the scene in simulation or the real world."""
        if self.simulator:
            self.ground = self.simulator.add_body(self.config.SIM.GROUND.PATH,
                                                  self.config.SIM.GROUND.POSE,
                                                  is_static=True,
                                                  name='ground')

            self.table_pose = Pose(self.config.SIM.TABLE.POSE)
            self.table_pose.position.z += np.random.uniform(
                *self.config.SIM.TABLE.HEIGHT_RANGE)
            self.table = self.simulator.add_body(self.config.SIM.TABLE.PATH,
                                                 self.table_pose,
                                                 is_static=True,
                                                 name='table')
            self.reset_camera()

    def reset_robot(self):
        """Reset the robot in simulation or the real world."""
        if self.simulator:
            self.robot = sawyer.SawyerSim(
                    simulator=self.simulator,
                    pose=self.config.SIM.ARM.POSE,
                    joint_positions=self.config.ARM.OFFSTAGE_POSITIONS,
                    config=self.config.SIM.ARM.CONFIG)
        else:
            self.robot = sawyer.SawyerReal()

    def build_camera(self,
                     height,
                     width,
                     intrinsics=None,
                     translation=None,
                     rotation=None,
                     intrinsics_noise=None,
                     translation_noise=None,
                     rotation_noise=None,
                     crop=None,
                     calibration_path=None):
        """Build the camera sensor."""
        if self.simulator:
            self.camera = BulletCamera(simulator=self.simulator,
                                       height=height,
                                       width=width,
                                       crop=crop)
        else:
            self.camera = Kinect2(height=height,
                                  width=width,
                                  crop=crop)

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

    def reset_camera(self):
        if self.simulator:
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

    def visualize(self, action, info):
        pass
