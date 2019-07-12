"""Samplers used by the policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.specs import tensor_spec  # NOQA

from robovat.policies import heuristics  # NOQA


class ActionSampler(object):
    """Uniformly samples action."""
    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config):
        self.action_spec = action_spec
        self.num_steps = config.NUM_STEPS

    def __call__(self, time_step, num_samples, seed):
        if self.num_steps is None:
            outer_dims = [num_samples]
        else:
            outer_dims = [self.num_steps, num_samples]

        return tensor_spec.sample_spec_nest(
            self.action_spec, seed=seed, outer_dims=outer_dims)


class ZSampler(object):
    """Uniformly samples latent action."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config):
        self.dim_z = config.DIM_Z
        self.num_steps = config.NUM_STEPS
        self.debug = config.DEBUG

    def __call__(self, time_step, num_samples, seed):
        if self.num_steps is None:
            outer_dims = [num_samples]
        else:
            outer_dims = [self.num_steps, num_samples]

        shape = outer_dims + [self.dim_z]
        z = tf.random_normal(shape, 0.0, 1.0, dtype=tf.float32)

        # if True:
        # if self.debug:
        #     print_op = tf.print(
        #         'z: ', z, '\n',
        #     )
        #     with tf.control_dependencies([print_op]):
        #         z = tf.identity(z)

        return z


class CSampler(object):
    """Uniformly samples latent action."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config):
        self.dim_c = config.DIM_C
        self.num_steps = config.NUM_STEPS
        self.debug = config.DEBUG

    def __call__(self, time_step, num_samples, seed):
        if self.num_steps is None:
            outer_dims = [num_samples]
        else:
            outer_dims = [self.num_steps, num_samples]

        shape = outer_dims + [self.dim_c]
        c = tf.random_normal(shape, 0.0, 1.0, dtype=tf.float32)

        return c


class HeuristicSampler(object):
    """Samples starting pose using heuristics."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config,
                 debug=False):
        self._action_spec = action_spec
        motion_shape = action_spec['motion'].shape
        assert len(motion_shape) == 1
        motion_size = int(motion_shape[0])

        self.num_steps = None

        self._sampler = heuristics.HeuristicSampler(
            use_primitive=config.ACTION.USE_PRIMITIVE,
            cspace_low=config.ACTION.CSPACE.LOW,
            cspace_high=config.ACTION.CSPACE.HIGH,
            motion_size=motion_size,
            translation_x=config.ACTION.MOTION.TRANSLATION_X,
            translation_y=config.ACTION.MOTION.TRANSLATION_Y,
            rotation=config.ACTION.MOTION.ROTATION,
            max_attemps=config.HEURISTICS.MAX_ATTEMPS,
            debug=debug)

    def __call__(self, time_step, num_samples, seed):
        position = tf.squeeze(time_step.observation['position'], 0)

        start, motion = tf.py_func(
            self._sampler.sample,
            [position, num_samples],
            [tf.float32, tf.float32])

        action = {
            'start': start,
            'motion': motion,
        }

        for key, value in action.items():
            shape = list(self._action_spec[key].shape)
            action[key] = tf.reshape(value, [num_samples] + shape)
            if self.num_steps is not None:
                action[key] = tf.expand_dims(action[key], 0)

        return action
