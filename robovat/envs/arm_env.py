"""The environment of robot arm.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.envs import robot_env
from robovat.math import Pose
from robovat.robots import sawyer


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
            # keep_list = [self.ground, self.table]
            #
            # for body_name, body in self.world.bodies.items():
            #     if body not in keep_list:
            #         self.world.remove_body(body_name)

            self.ground = self.simulator.add_body(self.config.SIM.GROUND.PATH,
                                                  self.config.SIM.GROUND.POSE,
                                                  is_static=True)

            self.table_pose = Pose(self.config.SIM.TABLE.POSE)
            self.table_pose.position.z += np.random.uniform(
                *self.config.SIM.TABLE.HEIGHT_RANGE)
            self.table = self.simulator.add_body(self.config.SIM.TABLE.PATH,
                                                 self.table_pose,
                                                 is_static=True)

    def reset_robot(self):
        """Reset the robot in simulation or the real world."""
        if self.simulator:
            self.robot = sawyer.SawyerSim(
                    simulator=self.simulator,
                    pose=self.config.SIM.ARM.POSE,
                    joint_positions=self.config.ARM.OFFSTAGE_POSITIONS)
        else:
            self.robot = sawyer.SawyerReal()
