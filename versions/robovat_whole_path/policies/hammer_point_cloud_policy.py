"""Grasping policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_step

from robovat.grasp import image_grasp_sampler
from robovat.networks import GQCNN
from robovat.policies import cem_policy
from robovat.policies import point_cloud_policy

from robovat.math import search_keypoints
from robovat.math import solve_actions

from robovat.math import Pose, get_transform

from cvae.build import forward

nest = tf.contrib.framework.nest


class AntipodalGraspSampler(object):
    """Samples random antipodal grasps from a depth image."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config,
                 debug=False):
        debug = debug and config.DEBUG

        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

        flat_action_spec = nest.flatten(self._action_spec)
        self._action_dtype = flat_action_spec[0].dtype
        self._action_shape = flat_action_spec[0].shape

        self._sampler = image_grasp_sampler.AntipodalDepthImageGraspSampler(
            friction_coef=config.SAMPLER.FRICTION_COEF,
            depth_grad_thresh=config.SAMPLER.DEPTH_GRAD_THRESH,
            depth_grad_gaussian_sigma=config.SAMPLER.DEPTH_GRAD_GAUSSIAN_SIGMA,
            downsample_rate=config.SAMPLER.DOWNSAMPLE_RATE,
            max_rejection_samples=config.SAMPLER.MAX_REJECTION_SAMPLES,
            crop=config.SAMPLER.CROP,
            min_dist_from_boundary=config.SAMPLER.MIN_DIST_FROM_BOUNDARY,
            min_grasp_dist=config.SAMPLER.MIN_GRASP_DIST,
            angle_dist_weight=config.SAMPLER.ANGLE_DIST_WEIGHT,
            depth_samples_per_grasp=config.SAMPLER.DEPTH_SAMPLES_PER_GRASP,
            min_depth_offset=config.SAMPLER.MIN_DEPTH_OFFSET,
            max_depth_offset=config.SAMPLER.MAX_DEPTH_OFFSET,
            depth_sample_window_height=(
                config.SAMPLER.DEPTH_SAMPLE_WINDOW_HEIGHT),
            depth_sample_window_width=config.SAMPLER.DEPTH_SAMPLE_WINDOW_WIDTH,
            gripper_width=config.GRIPPER_WIDTH,
            debug=debug)

    def __call__(self, time_step, num_samples, seed):
        observation = nest.map_structure(lambda x: tf.squeeze(x, 0),
                                         time_step.observation)
        depth = observation['depth']
        intrinsics = observation['intrinsics']
        grasps = tf.py_func(
            self._sampler.sample, [depth, intrinsics, num_samples], tf.float32)
        grasps = tf.reshape(
            grasps, [1, num_samples] + self._action_shape.as_list())
        return grasps


class Grasp4DofRandomPolicy(random_tf_policy.RandomTFPolicy):
    """Sample random antipodal grasps."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 policy_state_spec=(),
                 config=None,
                 debug=False):
        self._sampler = AntipodalGraspSampler(
            time_step_spec, action_spec, config, debug=True)
        self._num_samples = 1

        super(Grasp4DofRandomPolicy, self).__init__(
            time_step_spec,
            action_spec,
            policy_state_spec)

    def _action(self, time_step, policy_state, seed):
        actions = self._sampler(
            time_step,
            self._num_samples,
            seed)
        action = tf.squeeze(actions, 0)
        return policy_step.PolicyStep(action, policy_state)


class HammerPointCloudPolicy(point_cloud_policy.PointCloudPolicy):

    TARGET_REGION = {
        'x': 0.2,
        'y': 0.2,
        'z': 0.1,
        'roll': 0,
        'pitch': 0,
        'yaw': np.pi,
    }

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None,
                 debug=False):

        super(HammerPointCloudPolicy, self).__init__(
            time_step_spec,
            action_spec,
            config=config)

        self.table_pose = Pose(self.config.SIM.TABLE.POSE)
        pose = Pose.uniform(**self.TARGET_REGION)
        self.target_pose = get_transform(
            source=self.table_pose).transform(pose)

        self.env_collision_points = np.array(
            [[0.15, 0.15, 0.4], [0.2, 0.2, 0.4],
             [0.1, 0.0, 0.4], [0.1, 0.1, 0.4]],
            dtype=np.float32)

    def _concat_actions(self, actions, num_dof=4):
        actions = tf.expand_dims(
            tf.concat(
                [tf.reshape(action, [-1, num_dof])
                 for action in actions],
                axis=0), 0)
        return actions

    def _hammer_end_rot(self):
        return np.pi / 2

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
        point_cloud_tf = time_step.observation['point_cloud']

        g_kp, f_kp, c_kp = tf.py_func(search_keypoints,
                                      [point_cloud_tf],
                                      [tf.float32, tf.float32, tf.float32])

        action, score = forward(
            point_cloud_tf * scale, g_kp * scale)

        action = tf.expand_dims(action, 0)
        xyz, rx, ry, rz = tf.split(action,
                                   [3, 1, 1, 1], axis=1)
        action_4dof = tf.concat([xyz / scale, rz], axis=1)

        c = tf.reduce_mean(
            tf.squeeze(point_cloud_tf), axis=0)

        g_kp = tf.squeeze(g_kp)
        f_kp = tf.squeeze(f_kp)

        v_cf = tf.reshape(f_kp - c, [3])
        v_cg = tf.reshape(g_kp - c, [3])
        s = tf.sign(v_cg[0] * v_cf[1] -
                    v_cg[1] * v_cf[0])

        action_xy = tf.squeeze(action_4dof[:, :2])
        v_af = -action_xy + f_kp[:2]
        d = tf.linalg.norm(v_af)

        start_rz = tf.atan2(y=v_af[1], x=v_af[0])

        pre_start_pose = tf.concat([
            action_xy, tf.constant([0.4],
                                   dtype=tf.float32), [start_rz]],
                                   axis=0)

        start_pose = tf.concat([
            action_xy, tf.constant([0.4],
                                   dtype=tf.float32),
            -s * tf.constant([np.pi / 2],
                             dtype=tf.float32)],
            axis=0)

        c_kp = c_kp - action_4dof[:, :3]
        drz = -s * tf.constant(np.pi / 2) - start_rz
        rot_mat = self._rot_mat(drz)
        c_kp = tf.matmul(c_kp,
                         tf.transpose(rot_mat, [1, 0]))

        t_xyz = self.target_pose.position
        meta_pose = tf.add(
            tf.constant(
                [t_xyz.x + 0.05, t_xyz.y, 0, 0],
                dtype=tf.float32),
            tf.add(
                tf.constant([0, 1, 0, 0],
                            dtype=tf.float32) * d,
                tf.constant([0, 0, 0, -np.pi / 2],
                            dtype=tf.float32)) * s)

        pre_target_pose = meta_pose + tf.constant(
            [0, 0, 0.4, 0], dtype=tf.float32)

        action_task = solve_actions(
            start_pose, pre_target_pose,
            self.config.SIM.TASK.T - 3,
            c_kp, self.env_collision_points)

        target_pose = meta_pose + tf.constant(
            [0, 0, 0.2, 0], dtype=tf.float32)

        action_task = self._concat_actions(
            [start_pose, action_task,
             pre_target_pose, target_pose])

        delta_pose = tf.constant([[0, 0, 0, 1]],
                                 dtype=tf.float32) * (
            action_4dof[0, 3] - start_rz)
        action_task = action_task + delta_pose

        action = {'grasp': action_4dof,
                  'task': action_task}

        return policy_step.PolicyStep(action, policy_state)


class Grasp4DofCemPolicy(cem_policy.CemPolicy):
    """4-DoF grasping policy using CEM."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None,
                 debug=False):
        initial_sampler = AntipodalGraspSampler(
            time_step_spec, action_spec, config, debug=False)
        q_network = tf.make_template(
            'GQCNN',
            GQCNN,
            create_scope_now_=True,
            time_step_spec=time_step_spec,
            action_spec=action_spec)
        q_network = q_network()
        super(Grasp4DofCemPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            q_network=q_network,
            initial_sampler=initial_sampler,
            config=config)
