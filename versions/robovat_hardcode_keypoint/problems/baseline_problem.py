"""Model predictive controller problems.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from robovat.problems import mpc_problem
from robovat.utils.logging import logger  # NOQA

slim = tf.contrib.slim
framework = tf.contrib.framework
nest = tf.contrib.framework.nest
Reduction = tf.losses.Reduction


STATE_LOSS_WEIGHT = 1e2
ACTION_LOSS_WEIGHT = 1e0
VALID_LOSS_WEIGHT = 1e3
REG_LOSS_WEIGHT = 1e-1

REG_LOSS_CLIP_VALUE = 0.05
VALID_LOSS_CLIP_VALUE = 0.5


class VMPCProblem(mpc_problem.MPCProblem):
    """Latent ldmamics Model Problem."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training):
        """Initialize."""
        super(VMPCProblem, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training)

    def loss(self, targets, outputs):
        weights = targets['valid']
        tf.summary.histogram('valid', weights)

        z_kld = self.normal_kld(
            outputs['z'],
            outputs['z_mean'],
            outputs['z_stddev'],
            weights)

        state_loss = self.state_loss(
            targets['next_state'],
            outputs['pred_state'],
            weights)

        action_loss = self.action_loss(
            targets['action'],
            outputs['action'],
            weights)

        valid_loss = self.valid_loss(
            tf.expand_dims(targets['valid'], -1),
            outputs['valid_logit'])

        if 'encoded_next_state' in outputs:
            encoding_loss = self.encoding_loss(
                outputs['encoded_next_state'],
                outputs['pred_state'],
                weights)
        else:
            encoding_loss = 0.0

        # Summary.
        with tf.variable_scope('loss_breakdown'):
            tf.summary.scalar('z_kld', z_kld)
            tf.summary.scalar('state_loss', state_loss)
            tf.summary.scalar('action_loss', action_loss)
            tf.summary.scalar('valid_loss', valid_loss)
            tf.summary.scalar('encoding_loss', encoding_loss)

        return (
            z_kld
            + state_loss * STATE_LOSS_WEIGHT
            + action_loss * ACTION_LOSS_WEIGHT
            + valid_loss * VALID_LOSS_WEIGHT
            + encoding_loss
        )
    
    def add_summaries(self, targets, outputs):
        """Add summaries."""
        eval_ops = []

        with tf.variable_scope('z_all'):
            tf.summary.histogram('z', outputs['z'])
            tf.summary.histogram('z_mean', outputs['z_mean'])
            tf.summary.histogram('z_stddev', outputs['z_stddev'])

        eval_ops += self.add_state_summaries(targets, outputs)

        return eval_ops


class SectarProblem(mpc_problem.MPCProblem):
    """Latent ldmamics Model Problem."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training):
        """Initialize."""
        super(SectarProblem, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training)

    def loss(self, targets, outputs):
        weights = targets['valid']
        tf.summary.histogram('valid', weights)

        z_kld = self.normal_kld(
            outputs['z'],
            outputs['z_mean'],
            outputs['z_stddev'],
            weights)

        state_loss = self.state_loss(
            targets['next_state'],
            outputs['pred_state'],
            weights)

        action_loss = self.action_loss(
            targets['action'],
            outputs['action'],
            weights)

        if 'encoded_next_state' in outputs:
            encoding_loss = self.encoding_loss(
                outputs['encoded_next_state'],
                outputs['pred_state'],
                weights)
        else:
            encoding_loss = 0.0

        # Summary.
        with tf.variable_scope('loss_breakdown'):
            tf.summary.scalar('z_kld', z_kld)
            tf.summary.scalar('state_loss', state_loss)
            tf.summary.scalar('action_loss', action_loss)
            tf.summary.scalar('encoding_loss', encoding_loss)

        return (
            z_kld
            + state_loss * STATE_LOSS_WEIGHT
            + action_loss * ACTION_LOSS_WEIGHT
            + encoding_loss
        )
    
    def add_summaries(self, targets, outputs):
        """Add summaries."""
        eval_ops = []

        with tf.variable_scope('z_all'):
            tf.summary.histogram('z', outputs['z'])
            tf.summary.histogram('z_mean', outputs['z_mean'])
            tf.summary.histogram('z_stddev', outputs['z_stddev'])

        eval_ops += self.add_state_summaries(targets, outputs)

        return eval_ops
