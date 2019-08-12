"""The environment of robot arm.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.envs import robot_env
from robovat.math import Pose, get_transform
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


class HammerArmEnv(ArmEnv):
    """The environment of robot hammering."""

    TARGET_REGION = {
            'x': 0.2,
            'y': 0.15,
            'z': 0.1,
            'roll': 0,
            'pitch': 0,
            'yaw': 0,
            }

    def __init__(self,
                 observations,
                 reward_fns,
                 simulator=None,
                 config=None,
                 debug=False):
        """Initialize."""
        super(HammerArmEnv, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=simulator,
            config=config,
            debug=debug)

    def reset_scene(self):
        """Reset the scene in simulation or the real world."""
        if self.simulator:
            self.ground = self.simulator.add_body(self.config.SIM.GROUND.PATH,
                                                  self.config.SIM.GROUND.POSE,
                                                  is_static=True,
                                                  name='ground')

            self.table_pose = Pose(self.config.SIM.TABLE.POSE)
            self.table = self.simulator.add_body(self.config.SIM.TABLE.PATH,
                                                 self.table_pose,
                                                 is_static=True,
                                                 name='table')
            self._reset_task()
            self.reset_camera()

    def _reset_task(self):
        """Reset the task region.
        """
        # Sample and load a target object.
        if self.simulator:
            pose = Pose.uniform(**self.TARGET_REGION)
            peg_pose = get_transform(source=self.table_pose).transform(pose)
            slot_offset = np.array([-self.config.SIM.SLOT_OFFSET, 0, 0])

            slot_pose = Pose([peg_pose.position + slot_offset,
                              peg_pose.orientation])

            self.slot = self.simulator.add_body(
                    self.config.SIM.SLOT_PATH, 
                    slot_pose,
                    is_static=True, 
                    name='slot')
            self.target = self.simulator.add_body(
                    self.config.SIM.PEG_PATH, 
                    peg_pose, 
                    is_static=False,
                    name='peg')

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


class PushArmEnv(ArmEnv):
    """The environment of robot pushing."""

    TARGET_REGION = [{
            'x': 0.2,
            'y': 0.17,
            'z': 0.1,
            'roll': 0,
            'pitch': 0,
            'yaw': 0},
            {
            'x': 0.2,
            'y': 0.23,
            'z': 0.1,
            'roll': 0,
            'pitch': 0,
            'yaw': 0}]

    def __init__(self,
                 observations,
                 reward_fns,
                 simulator=None,
                 config=None,
                 debug=False):
        """Initialize."""
        super(PushArmEnv, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=simulator,
            config=config,
            debug=debug)

    def reset_scene(self):
        """Reset the scene in simulation or the real world."""
        if self.simulator:
            self.ground = self.simulator.add_body(self.config.SIM.GROUND.PATH,
                                                  self.config.SIM.GROUND.POSE,
                                                  is_static=True,
                                                  name='ground')

            self.table_pose = Pose(self.config.SIM.TABLE.POSE)
            self.table = self.simulator.add_body(self.config.SIM.TABLE.PATH,
                                                 self.table_pose,
                                                 is_static=True,
                                                 name='table')
            self._reset_task()
            self.reset_camera()

    def _reset_task(self):
        """Reset the task region.
        """
        # Sample and load a target object.
        if self.simulator:
            self.target = []

            for iregion, region in enumerate(self.TARGET_REGION):
                pose = Pose.uniform(**region)
                target_pose = get_transform(
                        source=self.table_pose).transform(pose)
                target = self.simulator.add_body(
                    self.config.SIM.TARGET_PATH,
                    target_pose, 
                    is_static=False,
                    name='target_{}'.format(iregion))
                self.target.append(target)


class ReachArmEnv(ArmEnv):
    """The environment of robot reaching."""

    TARGET_REGION = {
            'x': 0.25,
            'y': 0.25,
            'z': 0.10,
            'roll': 0,
            'pitch': 0,
            'yaw': np.pi/2}

    WALL_REGION = [{
            'x': 0.20,
            'y': 0.25,
            'z': 0.05,
            'roll': 0,
            'pitch': 0,
            'yaw': np.pi/2},
            {
            'x': 0.30,
            'y': 0.25,
            'z': 0.05,
            'roll': 0,
            'pitch': 0,
            'yaw': np.pi/2}]

    CEIL_REGION = {
            'x': 0.25,
            'y': 0.25,
            'z': 0.15,
            'roll': 0,
            'pitch': 0,
            'yaw': np.pi/2}

    def __init__(self,
                 observations,
                 reward_fns,
                 simulator=None,
                 config=None,
                 debug=False):
        """Initialize."""
        super(ReachArmEnv, self).__init__(
            observations=observations,
            reward_fns=reward_fns,
            simulator=simulator,
            config=config,
            debug=debug)

    def reset_scene(self):
        """Reset the scene in simulation or the real world."""
        if self.simulator:
            self.ground = self.simulator.add_body(self.config.SIM.GROUND.PATH,
                                                  self.config.SIM.GROUND.POSE,
                                                  is_static=True,
                                                  name='ground')

            self.table_pose = Pose(self.config.SIM.TABLE.POSE)
            self.table = self.simulator.add_body(self.config.SIM.TABLE.PATH,
                                                 self.table_pose,
                                                 is_static=True,
                                                 name='table')
            self._reset_task()
            self.reset_camera()

    def _reset_task(self):
        """Reset the task region.
        """
        # Sample and load a target object.
        if self.simulator:
            pose = Pose.uniform(**self.TARGET_REGION)
            target_pose = get_transform(source=self.table_pose).transform(pose)
            self.target = self.simulator.add_body(
                    self.config.SIM.TARGET_PATH, 
                    target_pose, 
                    is_static=False,
                    name='target')

            self.walls = []
            for iregion, region in enumerate(self.WALL_REGION):
                pose = Pose.uniform(**region)
                wall_pose = get_transform(
                        source=self.table_pose).transform(pose)
                wall = self.simulator.add_body(
                    self.config.SIM.WALL_PATH,
                    wall_pose, 
                    is_static=True,
                    name='wall_{}'.format(iregion))
                self.walls.append(wall)

            pose = Pose.uniform(**self.CEIL_REGION)
            ceil_pose = get_transform(source=self.table_pose).transform(pose)
            self.ceil = self.simulator.add_body(
                    self.config.SIM.CEIL_PATH, 
                    ceil_pose, 
                    is_static=True,
                    name='ceil')
