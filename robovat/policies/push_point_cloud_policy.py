"""Pushing policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.policies import policy_step
from robovat.policies import point_cloud_policy

from robovat.math import push_keypoints_heuristic
from robovat.math import solver_general

from robovat.math import Pose, get_transform

from keypoints.cvae.build import forward_keypoint
from keypoints.cvae.build import forward_grasp

nest = tf.contrib.framework.nest


class PushPointCloudPolicy(point_cloud_policy.PointCloudPolicy):
    """Predicts the pushing actions from point cloud"""
    
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
        """Initialization"""

        super(PushPointCloudPolicy, self).__init__(
            time_step_spec,
            action_spec,
            config=config)

        self.table_pose = Pose(self.TABLE_POSE)
        pose = Pose.uniform(**self.TARGET_REGION)
        self.target_pose = get_transform(
            source=self.table_pose).transform(pose)
        self.is_training = is_training

    def _concat_actions(self, actions, num_dof=4):
        """Concatenates the tool pose for each step."""
        actions = tf.expand_dims(
            tf.concat(
                [tf.reshape(action, [-1, num_dof])
                 for action in actions],
                axis=0), 0)
        return actions

    def _rot_mat(self, rz):
        """Computes the rotation matrix around the z axis.

        Args:
            rz: The rotation around z axis.

        Returns:
            mat: The rotation matrix.

        """
        zero = tf.constant(0.0,
                           dtype=tf.float32)
        one = tf.constant(1.0,
                          dtype=tf.float32)

        mat = tf.reshape(
            [[tf.cos(rz), -tf.sin(rz), zero],
             [tf.sin(rz), tf.cos(rz), zero],
             [zero, zero, one]], [3, 3])
        return mat

    def _keypoints_heuristic(self, point_cloud_tf):
        """Uses the heuristic policy to predict keypoints.
        
        Args:
            point_cloud_tf: The point cloud tensor.
            
        Returns:
            g_kp: The grasp point.
            f_kp: The function point.
            f_v: The vector pointing from the function point to effect point.
        """
        point_cloud_tf = tf.Print(
                point_cloud_tf, [], message='Using heuristic policy')

        g_kp, f_kp, f_v = tf.py_func(push_keypoints_heuristic,
                                     [point_cloud_tf],
                                     [tf.float32, tf.float32, tf.float32])
        return g_kp, f_kp, f_v

    def _keypoints_network(self, point_cloud_tf, scale=20):
        """Uses the neural network policy to predict keypoints.
        
        Args:
            point_cloud_tf: The point cloud tensor.
            scale: A constant coefficient that the point cloud coordinates 
                should be multiplied with to fit the input scale of the network.
            
        Returns:
            g_kp: The grasp point.
            f_kp: The function point.
            f_v: The vector pointing from the function point to effect point.
        """
        point_cloud_tf = tf.Print(
                point_cloud_tf, [], message='Using network policy')
        keypoints, f_v, _ = forward_keypoint(
                point_cloud_tf * scale,
                num_funct_vect=1,
                funct_on_hull=False)
        g_kp, f_kp = keypoints
        g_kp = g_kp / scale
        f_kp = f_kp / scale
        return g_kp, f_kp, f_v

    def _action(self,
                time_step,
                policy_state,
                seed, 
                scale=20):
        """Predicts the actions from visual observation.
        
        Args:
            time_step: A batch of timesteps.
            policy_state: The current policy state.
            seed: A random seed.
            scale: A constant coefficient that the point cloud coordinates 
                should be multiplied with to fit the input scale of the network.

        Returns:
            The grasp, action waypoints and the keypoints.
        """
        point_cloud_tf = time_step.observation['point_cloud']
        
        if self.is_training:
            g_kp, f_kp, f_v = self._keypoints_heuristic(point_cloud_tf)
        else:
            g_kp, f_kp, f_v = self._keypoints_network(point_cloud_tf)

        keypoints = tf.concat([g_kp, f_kp, f_v], axis=0)
        keypoints = tf.expand_dims(keypoints, axis=0)

        action, score = forward_grasp(
            point_cloud_tf * scale, g_kp * scale)

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
                                [target, force * 0.001, theta, d],
                                [tf.float32, tf.float32])

        g_rz = g_rz - start_rz + action_4dof[0, 3]

        target_force = tf.concat([
            force, tf.constant([0, 0], dtype=tf.float32)], axis=0)

        pre_overhead_pose = tf.concat([
            g_xy, tf.constant([0.40], dtype=tf.float32),
            [g_rz]],
            axis=0)
        overhead_pose = tf.concat([
            g_xy - force * 0.12, tf.constant([0.40], dtype=tf.float32),
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
            [target_force, pre_overhead_pose,
                overhead_pose, pre_target_pose, target_pose])

        action = {'grasp': action_4dof,
                  'task': action_task,
                  'keypoints': keypoints}

        return policy_step.PolicyStep(action, policy_state)

