"""Model predictive controller for the pushing environment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from robovat.networks import mpc
from robovat.networks import baseline
from robovat.networks import push_layers as layers
from robovat.utils.logging import logger  # NOQA

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


class PushMPC(mpc.MPC):

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(PushMPC, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training,
            config=config,
            name=name)

    def _encode_state(self, state, is_training):
        return layers.encode_state(state, is_training, self.config)

    def _forward_dynamics(self, state, action, is_training):
        return layers.forward_dynamics(state, action, is_training, self.config)


class PushVMPC(baseline.VMPC):

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(PushVMPC, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training,
            config=config,
            name=name)

    def _encode_state(self, state, is_training):
        return layers.encode_state(state, is_training, self.config)

    def _infer_z(self, state, action, is_training):
        return layers.infer_z_given_sa(
            state, action, is_training, self.config)

    def _predict_action(self, state, z, is_training):
        return layers.predict_action_given_z(state, z, is_training, self.config)

    def _forward_dynamics(self, state, action, is_training):
        return layers.forward_dynamics(state, action, is_training, self.config)


class PushSectar(baseline.Sectar):

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(PushSectar, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training,
            config=config,
            name=name)

    def _encode_state(self, state, is_training):
        return layers.encode_state(state, is_training, self.config)

    def _infer_z(self, state, next_state, is_training):
        # Our c is their z.
        return layers.infer_c(
            state, next_state, is_training, self.config)

    def _predict_action(self, state, z, is_training):
        return layers.predict_action_given_z(state, z, is_training, self.config)

    def _predict_transition(self, state, z, is_training):
        # Our c is their z.
        return layers.predict_transition(state, z, is_training, self.config)


class PushSMPC(mpc.SMPC):

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(PushSMPC, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training,
            config=config,
            name=name)

    def get_state_dist(self, a, b):
        dists = tf.norm(
            a['position'][..., :2] - b['position'][..., :2],
            axis=-1)
        return tf.reduce_sum(dists, axis=1)

    def _encode_state(self, state, is_training):
        return layers.encode_state(state, is_training, self.config)

    def _infer_c(self, state, next_state, is_training):
        return layers.infer_c(state, next_state, is_training, self.config)

    def _infer_z(self, state, action, c, is_training):
        return layers.infer_z(state, action, c, is_training, self.config)

    def _predict_transition(self, state, c, is_training):
        return layers.predict_transition(state, c, is_training, self.config)

    def _predict_action(self, state, z, c, is_training):
        return layers.predict_action(state, z, c, is_training, self.config)

    def _forward_dynamics(self, state, action, is_training):
        return layers.forward_dynamics(state, action, is_training, self.config)


class PushSMPCConcat(PushSMPC):
    def _predict_action(self, state, z, c, is_training):
        return layers.predict_action_concat(
            state, z, c, is_training, self.config)
