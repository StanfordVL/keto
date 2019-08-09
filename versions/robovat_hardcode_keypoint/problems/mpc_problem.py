"""Model predictive controller problems.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf

from robovat.problems import problem
from robovat.utils import loss_utils
from robovat.utils.logging import logger  # NOQA

slim = tf.contrib.slim
framework = tf.contrib.framework
nest = tf.contrib.framework.nest
Reduction = tf.losses.Reduction


STATE_LOSS_WEIGHT = 1e2
ACTION_LOSS_WEIGHT = 1e2
VALID_LOSS_WEIGHT = 1e1

# REG_LOSS_WEIGHT = 1e-1
# REG_LOSS_WEIGHT = 0.0
REG_LOSS_WEIGHT = 1e2

# REG_LOSS_CLIP_VALUE = None
REG_LOSS_CLIP_VALUE = 0.05
VALID_LOSS_CLIP_VALUE = 0.5


class MPCProblem(problem.Problem):
    """Latent ldmamics Model Problem."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training):
        """Initialize."""
        super(MPCProblem, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training)

    def normal_kld(self, z, z_mean, z_stddev, weights=1.0):
        return self._normal_kld(z, z_mean, z_stddev, weights)

    def valid_loss(self, targets, outputs, weights=1.0):
        return self._valid_loss(targets, outputs, weights)

    def state_loss(self, targets, outputs, weights=1.0, clip_value=None):
        return self._state_loss(targets, outputs, weights,
                                clip_value=clip_value)

    def action_loss(self, targets, outputs, weights=1.0):
        return self._action_loss(targets, outputs, weights)

    def encoding_loss(self, targets, outputs, weights=1.0):
        return self._encoding_loss(targets, outputs, weights)

    def loss(self, targets, outputs):
        weights = targets['valid']
        tf.summary.histogram('valid', weights)

        state_loss = self.state_loss(
            targets['next_state'],
            outputs['pred_state'],
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

        with tf.name_scope('loss_dynamics_breakdown'):
            tf.summary.scalar('state_loss', state_loss)
            tf.summary.scalar('valid_loss', valid_loss)
            tf.summary.scalar('encoding_loss', encoding_loss)

        return (
            state_loss * STATE_LOSS_WEIGHT
            + valid_loss * VALID_LOSS_WEIGHT
            + encoding_loss
        )
    
    def add_summaries(self, targets, outputs):
        """Add summaries."""
        eval_ops = []
        eval_ops += self.add_state_summaries(targets, outputs)
        eval_ops += self.add_action_summaries(targets, outputs)
        return eval_ops

    def add_state_summaries(self, targets, outputs):
        return []

    def add_action_summaries(self, targets, outputs):
        return []

    def _normal_kld(self, z, z_mean, z_stddev, weights=1.0,
                    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
        kld_array = (loss_utils.log_normal(z, z_mean, z_stddev) -
                     loss_utils.log_normal(z, 0.0, 1.0))
        return tf.losses.compute_weighted_loss(
            kld_array, weights, reduction=reduction)

    def _valid_loss(self, targets, outputs, weights=1.0):
        return tf.losses.sigmoid_cross_entropy(
            tf.cast(targets, tf.int64),
            outputs,
            weights)

    @abc.abstractmethod
    def _state_loss(self, targets, outputs, weights=1.0,
                    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
        raise NotImplementedError

    @abc.abstractmethod
    def _action_loss(self, targets, outputs, weights=1.0,
                     reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
        raise NotImplementedError

    @abc.abstractmethod
    def _encoding_loss(self, targets, outputs, weights=1.0,
                       reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
        raise NotImplementedError


class SMPCProblem(MPCProblem):
    """Latent ldmamics Model Problem."""

    def dynamics_loss(self, targets, outputs):
        with tf.variable_scope('loss_dynamics'):
            weights = targets['valid']

            state_loss = self.state_loss(
                targets['next_state'],
                outputs['pred_state'],
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

        with tf.name_scope('loss_dynamics_breakdown'):
            tf.summary.scalar('state_loss', state_loss)
            tf.summary.scalar('valid_loss', valid_loss)
            tf.summary.scalar('encoding_loss', encoding_loss)

        return (
            state_loss * STATE_LOSS_WEIGHT
            + valid_loss * VALID_LOSS_WEIGHT
            + encoding_loss
        )

    def vae_loss(self, targets, outputs):
        with tf.variable_scope('loss_vae'):
            weights = targets['valid']

            z_kld = self.normal_kld(
                outputs['z'],
                outputs['z_mean'],
                outputs['z_stddev'],
                weights)

            c_kld = self.normal_kld(
                outputs['c'],
                outputs['c_mean'],
                outputs['c_stddev'],
                weights)
    
            state_loss_high = self.state_loss(
                targets['next_state'],
                outputs['pred_state_high'],
                weights)

            action_loss = self.action_loss(
                targets['action'],
                outputs['action'],
                weights)

            if 'encoded_next_state' in outputs:
                encoding_loss_high = self.encoding_loss(
                    outputs['encoded_next_state'],
                    outputs['pred_state_high'],
                    weights)
            else:
                encoding_loss_high = 0.0

        with tf.name_scope('loss_vae_breakdown'):
            tf.summary.scalar('z_kld', z_kld)
            tf.summary.scalar('c_kld', c_kld)
            tf.summary.scalar('action_loss', action_loss)
            tf.summary.scalar('state_loss_high', state_loss_high)
            tf.summary.scalar('encoding_loss', encoding_loss_high)

        return (
            z_kld
            + c_kld
            + action_loss * ACTION_LOSS_WEIGHT
            + state_loss_high * STATE_LOSS_WEIGHT
            + encoding_loss_high
            )

    def reg_loss(self, targets, outputs):
        with tf.variable_scope('loss_reg'):
            weights = targets['valid']

            state_loss_gen = self.state_loss(
                outputs['pred_state_high_sg'],
                outputs['pred_state_gen'],
                weights,
                clip_value=REG_LOSS_CLIP_VALUE)

            valid_loss_gen = self.valid_loss(
               tf.ones_like(outputs['valid_logit_gen'], dtype=tf.int64),
               # outputs['valid_logit_gen'])
               tf.maximum(outputs['valid_logit_gen'], VALID_LOSS_CLIP_VALUE))

            if 'encoded_next_state' in outputs:
                encoding_loss_gen = self.encoding_loss(
                    outputs['encoded_next_state'],
                    outputs['pred_state_gen'],
                    weights)
            else:
                encoding_loss_gen = 0.0

        with tf.name_scope('loss_reg_breakdown'):
            tf.summary.scalar('state_loss_gen', state_loss_gen)
            tf.summary.scalar('valid_loss_gen', valid_loss_gen)
            tf.summary.scalar('encoding_loss_gen', encoding_loss_gen)

        return (
            0.0
            + state_loss_gen * STATE_LOSS_WEIGHT
            + valid_loss_gen * VALID_LOSS_WEIGHT
            + encoding_loss_gen
            )

    def add_summaries(self, targets, outputs):
        """Add summaries."""
        eval_ops = []

        tf.summary.histogram('valid', targets['valid'])

        with tf.variable_scope('vae_z'):
            tf.summary.histogram('z', outputs['z'])
            tf.summary.histogram('z_mean', outputs['z_mean'])
            tf.summary.histogram('z_stddev', outputs['z_stddev'])

        with tf.variable_scope('vae_c'):
            tf.summary.histogram('c', outputs['c'])
            tf.summary.histogram('c_mean', outputs['c_mean'])
            tf.summary.histogram('c_stddev', outputs['c_stddev'])

        eval_ops += self.add_state_summaries(targets, outputs)
        eval_ops += self.add_action_summaries(targets, outputs)

        with tf.variable_scope('valid_classification'):
            target_label = tf.expand_dims(targets['valid'], -1)
            output_prob = outputs['valid_prob']
            output_label = tf.cast(tf.greater(output_prob, 0.5), tf.int64)

            names_to_values, names_to_updates = (
                slim.metrics.aggregate_metric_map({
                        'valid/AUC': slim.metrics.streaming_auc(
                            output_prob, target_label),
                        'valid/accuracy': slim.metrics.streaming_accuracy(
                            output_label, target_label),
                        'valid/precision': slim.metrics.streaming_precision(
                            output_label, target_label),
                        'valid/recall': slim.metrics.streaming_recall(
                            output_label, target_label),
                }))

            for metric_name, metric_value in names_to_values.items():
                tf.summary.scalar(metric_name, metric_value)

            tf.summary.histogram('targets_valid', target_label)
            tf.summary.histogram('outputs_valid', output_label)

            eval_ops += names_to_updates.values()

        return eval_ops

    def get_train_op(self,
                     network,
                     batch,
                     learning_rate=0.0001,
                     clip_gradient_norm=25.0,
                     ):
        with tf.variable_scope('model/forward_dynamics') as dyn_scope:
            pass
        with tf.variable_scope('model/encode_state') as enc_scope:
            pass
        with tf.variable_scope('model/predict_action') as reg_scope:
            pass

        logger.info('Forwarding network...')
        outputs = network.forward(batch)

        logger.info('Building the loss function...')
        dynamics_loss = self.dynamics_loss(batch, outputs)
        vae_loss = self.vae_loss(batch, outputs)
        reg_loss = self.reg_loss(batch, outputs)

        logger.info('Building the train_op...')
        slim.get_or_create_global_step()
        learning_rate = self.get_learning_rate(learning_rate)

        dyn_opt = tf.train.AdamOptimizer(learning_rate)
        dyn_vars = (framework.get_trainable_variables(dyn_scope) +
                    framework.get_trainable_variables(enc_scope))
        dyn_train_op = slim.learning.create_train_op(
            dynamics_loss,
            dyn_opt,
            variables_to_train=dyn_vars,
            clip_gradient_norm=clip_gradient_norm,
            summarize_gradients=True)

        vae_opt = tf.train.AdamOptimizer(learning_rate)
        vae_vars = [
            var
            for var in framework.get_trainable_variables()
            if var not in dyn_vars]
        vae_train_op = slim.learning.create_train_op(
            vae_loss,
            vae_opt,
            variables_to_train=vae_vars,
            clip_gradient_norm=clip_gradient_norm,
            summarize_gradients=True)

        reg_opt = tf.train.AdamOptimizer(learning_rate)
        reg_vars = framework.get_trainable_variables(reg_scope)
        reg_train_op = slim.learning.create_train_op(
            reg_loss * REG_LOSS_WEIGHT,
            reg_opt,
            variables_to_train=reg_vars,
            clip_gradient_norm=clip_gradient_norm,
            summarize_gradients=True)

        print('#dyn_vars: %d, #vae_vars: %d, #reg_vars: %d'
              % (len(dyn_vars), len(vae_vars), len(reg_vars)))
        train_op = tf.group(dyn_train_op, vae_train_op, reg_train_op)
        # train_op = tf.group(dyn_train_op, vae_train_op)

        logger.info('Adding summaries...')
        tf.summary.scalar('loss', dynamics_loss + vae_loss)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.name_scope('variables'):
            for var in variables:
                tf.summary.histogram(var.name, var)
        eval_ops = self.add_summaries(batch, outputs)
        train_op = tf.group(train_op, *eval_ops)

        return train_op
