"""Reward function of the environments.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.envs.reward_fns import reward_fn
from robovat.utils.logging import logger


class PushReward(reward_fn.RewardFn):
    """Reward function of the environments."""
    
    def __init__(self,
                 name,
                 graspable_name=None,
                 target_name=None,
                 terminate_after_grasp=True,
                 streaming_length=1000):
        """Initialize."""
        self.name = name 
        self.graspable_name = graspable_name
        self.target_name = target_name
        self.terminate_after_grasp = terminate_after_grasp
        self.streaming_length = streaming_length

        self.env = None
        self.target = None
        self.graspable = None
        self.history = []

    def on_episode_start(self):
        """Called at the start of each episode."""
        self.target = []
        self.target_pose_init = []

        for target_name in self.target_name:
            target = self.env.simulator.bodies[target_name]
            target_pose_init = np.array(
                target.pose.position)
            self.target.append(target)
            self.target_pose_init.append(target_pose_init)

        self.graspable = self.env.simulator.bodies[self.graspable_name]
        self.env.timeout = False
        self.env.grasp_cornercase = False

    def get_reward(self):
        """Returns the reward value of the current step."""
        if self.env.simulator:
            all_trans = []
            for target, pose_init in zip(self.target, self.target_pose_init):
                # self.env.simulator.wait_until_stable(target)
                target_pose = np.array(target.pose.position)
                trans = target_pose[0] - pose_init[0]
                trans_z = abs(target_pose[2] - pose_init[2])
                suc = trans > 0.05 or trans_z > 0.2
                all_trans.append(suc)

            success = np.all(all_trans)

        else:
            raise NotImplementedError

        if self._check_cornercase():
            logger.debug('Ignore cornercase')
            success = -1
        else:
            self._update_history(success)
            success_rate = np.mean(self.history or [-1])
            logger.debug('Hammer Success: %r, Success Rate %.3f',
                         success, success_rate)
        return success, self.terminate_after_grasp

    def _check_cornercase(self):
        is_cnc = self.env.timeout or self.env.grasp_cornercase
        return is_cnc

    def _update_history(self, success):
        self.history.append(success)

        if len(self.history) > self.streaming_length:
            self.history = self.history[-self.streaming_length:]
