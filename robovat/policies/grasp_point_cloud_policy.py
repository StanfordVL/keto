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

from keypoints.cvae.build import forward_grasp

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


class Grasp4DofPointCloudPolicy(point_cloud_policy.PointCloudPolicy):

    def __init__(self, 
                 time_step_spec,
                 action_spec,
                 config=None,
                 debug=False):

        super(Grasp4DofPointCloudPolicy, self).__init__(
                time_step_spec,
                action_spec,
                config=config)
        
    def _action(self, 
                time_step, 
                policy_state, 
                seed, 
                scale=20):
        point_cloud_tf = time_step.observation['point_cloud']
        action, score = forward_grasp(point_cloud_tf * scale)
        action = tf.expand_dims(action, 0)
        xyz, rx, ry, rz = tf.split(action, 
                [3, 1, 1, 1], axis=1)
        action_4dof = tf.concat([xyz / scale, rz], axis=1)
        return policy_step.PolicyStep(action_4dof, policy_state)


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
