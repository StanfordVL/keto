"""The parent class for Sawyer robot environments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time  # NOQA

import numpy as np

from robovat.gym import Env
from robovat.math import Pose
from robovat.robot import sawyer
from robovat.simulation import World
from robovat.simulation.sensor import BulletCameraSensor
from robovat.sensor import Kinect2Sensor
from robovat.utils.logging import logger


class SawyerEnv(Env):
    """The parent class for Sawyer robot environments.
    """
    # Configuration of the basic scene.
    GROUND_URDF = os.path.join('planes', 'plane.urdf')
    GROUND_POSE = [[0, 0, -0.9], [0, 0, 0]]
    # TABLE_URDF = os.path.join('tables', 'svl_table', 'table.urdf')
    TABLE_URDF = os.path.join('tables', 'svl_table', 'table_grasp.urdf')
    TABLE_MEAN_POSE = [[0.6, 0, -0.9], [0, 0, np.pi / 2]]
    TABLE_HEIGHT_RANGE = (0.9, 0.9)
    ROBOT_POSE = [[0, 0, 0], [0, 0, 0]]

    # Configuration of the camera.
    # TODO(kuanfang): The camera translation does not match the real world yet.
    CAMERA_DISTANCE = 1.0
    CAMERA_INTRINSICS_PATH = os.path.join('data', 'calibration', 'kinect',
                                          'IR_intrinsics.npy')
    CAMERA_ROTATION = np.array([-np.pi, 0, 0])
    CAMERA_ROTATION_RANGE = np.array([np.pi / 70, np.pi / 70, np.pi / 70])
    CAMERA_TRANSLATION = np.array([-0.70, 0.00, 1.15])
    CAMERA_TRANSLATION_RANGE = np.array([0.05, 0.1, 0.03])
    # Predefined joint positions of arm.
    OUT_OF_VIEW_POSITIONS = [-1.5, -1.26, 0.00, 1.98, 0.00, 0.85, 3.3161]
    OVERHEAD_POSITIONS = [-0.73, -1.13, 0.82, 1.51, -0.33, 1.34, 1.87]

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
        Env.__init__(self,
                     is_simulation=is_simulation,
                     data_dir=data_dir,
                     key=key,
                     cfg=cfg,
                     debug=debug,
                     camera=camera)

        if self.is_simulation:
            self.world = World(key=key, debug=debug)
            self.robot_pose = self.ROBOT_POSE

            self.camera = BulletCameraSensor(
                    world=self.world,
                    robot_pose=self.robot_pose,
                    distance=self.CAMERA_DISTANCE)
            self._randomize_camera()

        else:
            self.robot_pose = self.ROBOT_POSE

            self.camera = Kinect2Sensor()
            self.camera.load_calibration('data/calibration/kinect/')
            self.camera.start()

    def _reset_robot_and_scene(self):
        """Reset the robot.

        Returns:
            The observation.
        """
        if self.is_simulation:
            # Reset the simulation.
            logger.debug('Resetting the world...')
            self.world.reset()
            self._randomize_camera()
            self.graspable = None

            # Start the simulation.
            self.world.start()

            # Build the robot.
            logger.debug('Building the robot...')
            self.robot = sawyer.SawyerSim(self.world,
                                          self.ROBOT_POSE,
                                          self.OUT_OF_VIEW_POSITIONS)

            # Build the scene.
            logger.debug('Building the scene...')
            self._build_scene()
        else:
            # Build the robot.
            self.robot = sawyer.SawyerReal()

        self.num_episodes += 1

    def _randomize_camera(self):
        """Randomlly set the camera calibration.
        """
        K = np.load(self.CAMERA_INTRINSICS_PATH, encoding='latin1')
        rotation = (self.CAMERA_ROTATION +
                    np.random.uniform(-self.CAMERA_ROTATION_RANGE,
                                      self.CAMERA_ROTATION_RANGE))
        translation = (self.CAMERA_TRANSLATION +
                       np.random.uniform(-self.CAMERA_TRANSLATION_RANGE,
                                         self.CAMERA_TRANSLATION_RANGE))
        self.camera.set_calibration(K, rotation, translation)

    def _add_body(self, path, pose=None, scale=1.0, is_static=False):
        """Add a body to the scene.
        """
        assert self.is_simulation, 'This function is only used in simulation.'

        if path.startswith('/'):
            full_path = path
        else:
            full_path = os.path.join(
                    self.data_dir, 'simulation', path)

        if pose is None:
            pose = [[0, 0, 0], [0, 0, 0]]

        body = self.world.add_body(filename=full_path, pose=pose, scale=scale,
                                   is_static=is_static)

        return body

    def _build_scene(self):
        """Build the basic scene.

        The scene is almost identical with our real-world setting in the
        Stanford Vision Lab. The basic scene contains the ground and the table.
        The basic scene is the same for every tasks on this platform.
        """
        assert self.is_simulation, 'This function is only used in simulation.'
        logger.debug('Building the basic scene...')

        self.ground = self._add_body(
                self.GROUND_URDF, self.GROUND_POSE, is_static=True)

        self.table_pose = Pose(self.TABLE_MEAN_POSE[0],
                               self.TABLE_MEAN_POSE[1])
        table_height = np.random.uniform(*self.TABLE_HEIGHT_RANGE)
        self.table_pose.position += np.array([0, 0, table_height])
        self.table = self._add_body(self.TABLE_URDF, self.table_pose,
                                    is_static=True)

    def _clear_scene(self):
        """Remove all bodies except for the robot and the basic scene.
        """
        assert self.is_simulation, 'This function is only used in simulation.'

        for body_name, body in self.world.bodies.items():
            if body not in [self.ground, self.table, self.robot.arm,
                            self.robot.base]:
                self.world.remove_body(body_name)

    def _wait_until_stable(self,
                           body,
                           linear_velocity_threshold=0.005,
                           angular_velocity_threshold=0.005,
                           check_after_steps=100,
                           min_stable_steps=100,
                           max_steps=2000):
        """Wait until the objects are stable.

        Args:
            body: List or instance of Body to check.
        """
        assert self.is_simulation, 'This function is only used in simulation.'
        logger.debug('Waiting for objects to be stable...')

        if isinstance(body, (list, tuple)):
            body_list = body
        else:
            body_list = [body]

        num_steps = 0
        num_stable_steps = 0

        while(1):
            self.world.step()
            num_steps += 1

            if num_steps < check_after_steps:
                continue

            # Check if all bodies are stable.
            all_stable = True
            for b in body_list:
                is_stable = self.world.check_stable(
                        b,
                        linear_velocity_threshold,
                        angular_velocity_threshold)

                if not is_stable:
                    all_stable = False
                    break

            if all_stable:
                num_stable_steps += 1

            if (num_stable_steps >= min_stable_steps or
                    num_steps >= max_steps):
                break

    @property
    def info(self):
        return {
                'camera_K': self.camera.K,
                'camera_rotation': self.camera.rotation,
                'camera_translation': self.camera.translation,
                }
