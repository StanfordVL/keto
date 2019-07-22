"""Reward function of the environments.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.envs.reward_fns import reward_fn
from robovat.utils.logging import logger


class HammerReward(reward_fn.RewardFn):
    """Reward function of the environments."""
    
    def __init__(self,
                 name,
                 target_name,
                 terminate_after_grasp=True,
                 streaming_length=1000):
        """Initialize."""
        self.name = name 
        self.target_name = target_name
        self.terminate_after_grasp = terminate_after_grasp
        self.streaming_length = streaming_length

        self.env = None
        self.target = None

        self.history = []

    def on_episode_start(self):
        """Called at the start of each episode."""
        self.target = self.env.simulator.bodies[self.target_name]
        self.target_pose_init = np.array(
                self.target.pose.position)

    def get_reward(self):
        """Returns the reward value of the current step."""
        if self.env.simulator:
            self.env.simulator.wait_until_stable(self.target)
            target_pose = np.array(self.target.pose.position)
            success = np.amax(
                    np.abs(target_pose - self.target_pose_init)) > 0.01
        else:
            raise NotImplementedError

        self._update_history(success)
        success_rate = np.mean(self.history or [-1])
        logger.debug('Hammer Success: %r, Success Rate %.3f',
                     success, success_rate)

        return success, self.terminate_after_grasp

    def _update_history(self, success):
        self.history.append(success)

        if len(self.history) > self.streaming_length:
            self.history = self.history[-self.streaming_length:]