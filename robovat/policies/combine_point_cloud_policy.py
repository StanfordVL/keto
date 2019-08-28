"""Grasping policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.policies import policy_step

from robovat.policies import point_cloud_policy

from robovat.math import combine_keypoints_heuristic
from robovat.math import solver_general

from robovat.math import Pose, get_transform

from keypoints.cvae.build import forward_keypoint
from keypoints.cvae.build import forward_grasp

nest = tf.contrib.framework.nest


class CombinePointCloudPolicy(point_cloud_policy.PointCloudPolicy):

    TARGET_REGION = {
        'x': 0.20,
        'y': 0.25,
        'z': 0.10,
        'roll': 0,
        'pitch': 0,
        'yaw': np.pi/2,
    }

    TABLE_POSE = [
            [0.6, 0, 0.0], 
            [0, 0, 0]]

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None,
                 debug=False,
                 is_training=True):

        super(CombinePointCloudPolicy, self).__init__(
            time_step_spec,
            action_spec,
            config=config)

        self.table_pose = Pose(self.TABLE_POSE)
        pose = Pose.uniform(**self.TARGET_REGION)
        self.target_pose = get_transform(
            source=self.table_pose).transform(pose)

        self.is_training = is_training

    def _concat_actions(self, actions, num_dof=4):
        actions = tf.expand_dims(
            tf.concat(
                [tf.reshape(action, [-1, num_dof])
                 for action in actions],
                axis=0), 0)
        return actions

    def _rot_mat(self, rz):
        zero = tf.constant(0.0,
                           dtype=tf.float32)
        one = tf.constant(1.0,
                          dtype=tf.float32)

        mat = tf.reshape(
            [[tf.cos(rz), -tf.sin(rz), zero],
             [tf.sin(rz), tf.cos(rz), zero],
             [zero, zero, one]], [3, 3])
        return mat
        
    def _action(self,
                time_step,
                policy_state,
                seed, 
                scale=20):

        point_cloud = time_step.observation['point_cloud']
        grasp_point, func_point, func_vect = tf.py_func(
                combine_keypoints_heuristic,
                [point_cloud], [tf.float32, tf.float32, tf.float32])

        keypoints = tf.concat([grasp_point, func_point, func_vect], axis=0)
        keypoints = tf.expand_dims(keypoints, axis=0)

        action_4dof = tf.zeros(shape=[1, 4], dtype=tf.float32)
        action_task = tf.zeros(shape=[1, 1, 3], dtype=tf.float32)

        action = {'grasp': action_4dof,
                  'task': action_task,
                  'keypoints': keypoints}

        return policy_step.PolicyStep(action, policy_state)

