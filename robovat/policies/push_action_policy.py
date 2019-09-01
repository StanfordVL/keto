"""Grasping policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.policies import policy_step
from robovat.policies import point_cloud_policy

from robovat.math import Pose, get_transform
from keypoints.cvae.build import forward_grasp
from keypoints.cvae.build import forward_action

nest = tf.contrib.framework.nest


class PushActionPolicy(point_cloud_policy.PointCloudPolicy):

    TARGET_REGION = {
        'x': 0.2,
        'y': 0.1,
        'z': 0.1,
        'roll': 0,
        'pitch': 0,
        'yaw': 0,
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

        super(PushActionPolicy, self).__init__(
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

    def _keypoints_network(self, point_cloud_tf, scale=20):
        actions, score = forward_action(point_cloud_tf * scale)
        grasp, trans, rot = actions
        g_kp = grasp / scale
        g_xy = trans / scale
        g_rz = tf.atan2(rot[:, 1], rot[:, 0])
        return g_kp, g_xy, g_rz

    def _action(self,
                time_step,
                policy_state,
                seed,
                scale=20):
        point_cloud_tf = time_step.observation['point_cloud']
        g_kp, g_xy, g_rz = self._keypoints_network(point_cloud_tf)

        keypoints = tf.zeros(shape=[1, 3, 3], dtype=tf.float32)

        action, score = forward_grasp(
            point_cloud_tf * scale, g_kp * scale)

        action = tf.expand_dims(action, 0)
        xyz, rx, ry, rz = tf.split(action,
                                   [3, 1, 1, 1], axis=1)
        action_4dof = tf.concat([xyz / scale, rz], axis=1)

        g_kp = tf.squeeze(g_kp)
        g_xy = tf.squeeze(g_xy)
        g_rz = tf.squeeze(g_rz)
        action_4dof = tf.concat([g_kp[:2], action_4dof[0, 2:]], axis=0)
        action_4dof = tf.expand_dims(action_4dof, axis=0)

        g_xy = g_xy + self.target_pose.position[:2]
        g_rz = g_rz + action_4dof[0, 3]

        trz = self.target_pose.euler[2]
        force = tf.constant([np.cos(trz), np.sin(trz)],
                            dtype=tf.float32)

        target_force = tf.concat([
            force, tf.constant([0, 0], dtype=tf.float32)], axis=0)

        overhead_pose = tf.concat([
            g_xy - force * 0.18, tf.constant([0.40], dtype=tf.float32),
            [g_rz]],
            axis=0)
        pre_target_pose = tf.concat([
            g_xy - force * 0.10, tf.constant([0.18], dtype=tf.float32),
            [g_rz]],
            axis=0)
        target_pose = tf.concat([
            g_xy + force * 0.05, tf.constant([0.18], dtype=tf.float32),
            [g_rz]],
            axis=0)

        action_task = self._concat_actions(
            [target_force, overhead_pose, pre_target_pose, target_pose])

        action = {'grasp': action_4dof,
                  'task': action_task,
                  'keypoints': keypoints}

        return policy_step.PolicyStep(action, policy_state)

