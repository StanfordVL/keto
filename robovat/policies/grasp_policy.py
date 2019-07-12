"""Grasping policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_step

from robovat.grasp import image_grasp_sampler
from robovat.policies import cem_policy

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
            boundary=config.SAMPLER.BOUNDARY,
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

    def __call__(self, observation, policy_state, num_samples):
        depth = tf.squeeze(observation['depth'], 0)
        intrinsics = tf.squeeze(observation['intrinsics'], 0)
        grasps = tf.py_func(
            self._sampler.sample, [depth, intrinsics, num_samples], tf.float32)
        grasps = tf.reshape(
            grasps, [num_samples] + self._action_shape.as_list())
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
            time_step_spec, action_spec, config)
        self._num_samples = 1

        super(Grasp4DofRandomPolicy, self).__init__(
            time_step_spec,
            action_spec,
            policy_state_spec)

    def _action(self, time_step, policy_state, seed):

        action = self._sampler(
            time_step.observation,
            policy_state,
            self._num_samples)

        return policy_step.PolicyStep(action, policy_state)


class Grasp4DofCemPolicy(cem_policy.CemPolicy):
    """4-DoF grasping policy using CEM."""

    def __init__(self,
                 time_step_spec=None,
                 action_spec=None,
                 critic_network=None,
                 encoding_network=None,
                 num_samples=64,
                 num_elites=6,
                 num_iterations=3,
                 config=None,
                 debug=False):

        initial_sampler = AntipodalGraspSampler(
            time_step_spec, action_spec, config)

        super(Grasp4DofCemPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=None,
            critic_network=critic_network,
            encoding_network=encoding_network,
            initial_sampler=initial_sampler,
            num_samples=num_samples,
            num_elites=num_elites,
            num_iterations=num_iterations)
