"""Grasping policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import tensorflow as tf
from tf_agents.policies import policy_step

from robovat.policies import point_cloud_policy

from robovat.math import reach_keypoints_heuristic
from robovat.math import solver_general

from robovat.math import Pose, get_transform
from robovat.math import match_keypoints

from keypoints.cvae.build import forward_keypoint
from keypoints.cvae.build import forward_grasp

nest = tf.contrib.framework.nest


class ReachMatchPolicy(point_cloud_policy.PointCloudPolicy):

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

    DATA_PATH = './keypoints/utils/baseline/data_reach.hdf5'

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None,
                 debug=False,
                 is_training=True):

        super(ReachMatchPolicy, self).__init__(
            time_step_spec,
            action_spec,
            config=config)

        self.table_pose = Pose(self.TABLE_POSE)
        pose = Pose.uniform(**self.TARGET_REGION)
        self.target_pose = get_transform(
            source=self.table_pose).transform(pose)

        self.is_training = is_training
        f = h5py.File(self.DATA_PATH, 'r')
        self.point_cloud_data = f['pos_point_cloud']
        self.keypoints_data = f['pos_keypoints']

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
        
    def _keypoints_match(self, point_cloud_tf, scale=20):
        [g_kp, f_kp, f_v
         ] = tf.py_func(
            match_keypoints,
            [point_cloud_tf * scale, 
                self.point_cloud_data, self.keypoints_data],
            [tf.float32, tf.float32, tf.float32])
        g_kp = g_kp / scale
        f_kp = f_kp / scale
        return g_kp, f_kp, f_v

    def _action(self,
                time_step,
                policy_state,
                seed, 
                scale=20):
        point_cloud_tf = time_step.observation['point_cloud']
        g_kp, f_kp, f_v = self._keypoints_match(point_cloud_tf)

        keypoints = tf.concat([g_kp, f_kp, f_v], axis=0)
        keypoints = tf.expand_dims(keypoints, axis=0)

        action, score = forward_grasp(
            point_cloud_tf * scale, 
            g_kp * scale,
            num_samples=128)

        action = tf.expand_dims(action, 0)
        xyz, rx, ry, rz = tf.split(action,
                                   [3, 1, 1, 1], axis=1)
        action_4dof = tf.concat([xyz / scale, rz], axis=1)

        g_kp = tf.squeeze(g_kp)
        f_kp = tf.squeeze(f_kp)
        f_v = tf.squeeze(f_v)

        action_4dof = tf.concat([g_kp[:2], action_4dof[0, 2:]], axis=0)
        action_4dof = tf.expand_dims(action_4dof, axis=0)

        v_fg = g_kp - f_kp
        theta = tf.add(tf.atan2(f_v[1], f_v[0]),
                       -tf.atan2(v_fg[1], v_fg[0]))

        action_xy = tf.squeeze(action_4dof[:, :2])
        v_af = -action_xy + f_kp[:2]
        d = tf.linalg.norm(v_af)

        start_rz = tf.atan2(y=v_af[1], x=v_af[0])

        tx, ty, tz = self.target_pose.position

        trz = self.target_pose.euler[2]
        target = tf.constant([tx, ty], dtype=tf.float32)
        force = tf.constant([np.cos(trz), np.sin(trz)],
                            dtype=tf.float32)

        g_xy, g_rz = tf.py_func(solver_general,
                                [target, force * 0.01, theta, d],
                                [tf.float32, tf.float32])

        g_rz = g_rz - start_rz + action_4dof[0, 3]

        target_force = tf.concat([
            force, tf.constant([0, 0], dtype=tf.float32)], axis=0)


        pre_pre_pre_target_pose = tf.concat([
            g_xy - force * 0.25, tf.constant([0.4], dtype=tf.float32),
            [g_rz]],
            axis=0)

        pre_pre_target_pose = tf.concat([
            g_xy - force * 0.25, tf.constant([0.20], dtype=tf.float32),
            [g_rz]],
            axis=0)

        pre_target_pose = tf.concat([
            g_xy - force * 0.12, tf.constant([0.18], dtype=tf.float32),
            [g_rz]],
            axis=0)

        reach_distance = 0.0 if self.is_training else 0.1 * force

        target_pose = tf.concat([
            g_xy + reach_distance, tf.constant([0.18], dtype=tf.float32),
            [g_rz]],
            axis=0)

        action_task = self._concat_actions(
            [target_force, pre_pre_pre_target_pose, 
                pre_pre_target_pose, pre_target_pose, target_pose])

        action = {'grasp': action_4dof,
                  'task': action_task,
                  'keypoints': keypoints}

        return policy_step.PolicyStep(action, policy_state)

