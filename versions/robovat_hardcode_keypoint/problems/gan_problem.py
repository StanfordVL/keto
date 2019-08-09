"""Model predictive controller problems.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.contrib.gan.python import namedtuples as gan_namedtuples  # NOQA

from robovat.problems import mpc_problem
from robovat.utils.logging import logger  # NOQA

slim = tf.contrib.slim
framework = tf.contrib.framework
nest = tf.contrib.framework.nest
tfgan = tf.contrib.gan
Reduction = tf.losses.Reduction


STATE_LOSS_WEIGHT = 1e2
ACTION_LOSS_WEIGHT = 1e2
VALID_LOSS_WEIGHT = 1e1
REG_LOSS_WEIGHT = 1.0
GRADIENT_PENALTY_WEIGHT = 1e4


class GMPCProblem(mpc_problem.SMPCProblem):
    """MPC problem using GAN."""

    def reg_loss(self, targets, outputs):
        with tf.variable_scope('loss_reg'):
            weights = targets['valid']

            state_loss = self.state_loss(
                outputs['pred_state_high_sg'],
                outputs['pred_state_low'],
                weights)

            valid_loss = self.valid_loss(
               tf.ones_like(outputs['valid_logit_low'], dtype=tf.int64),
               outputs['valid_logit_low'])

        with tf.name_scope('loss_reg_breakdown'):
            tf.summary.scalar('state_loss', state_loss)
            tf.summary.scalar('valid_loss', valid_loss)

        return (
            state_loss * STATE_LOSS_WEIGHT
            )

    def add_summaries(self, targets, outputs):
        """Add summaries."""
        eval_ops = []

        tf.summary.histogram('valid', targets['valid'])

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
                     learning_rate,
                     clip_gradient_norm=-1):

        slim.get_or_create_global_step()
        learning_rate = self.get_learning_rate(learning_rate)

        logger.info('Forwarding network...')
        outputs = network.forward(batch)

        ####
        # Dynamics.
        ####
        logger.info('Building dynamics model...')

        dyn_loss = self.dynamics_loss(batch, outputs)
        dyn_opt = tf.train.RMSPropOptimizer(learning_rate)
        dyn_train_op = slim.learning.create_train_op(
            dyn_loss,
            dyn_opt,
            clip_gradient_norm=clip_gradient_norm,
            summarize_gradients=True)

        ####
        # High-level GAN.
        ####
        logger.info('Building high-level GAN...')

        high_train_ops = self.get_gan_train_ops(
            gen_scope='model/predict_transition',
            dis_scope='model/discriminate_state',
            dis_logit_real=outputs['dis_logit_real_high'],
            dis_logit_gen=outputs['dis_logit_gen_high'],
            gradient_penalty=outputs['gradient_penalty_high'],
            scope='gan_high')

        ####
        # Low-level GAN.
        ####
        logger.info('Building low-level GAN...')

        reg_loss = self.reg_loss(batch, outputs)
        reg_loss = 0.0

        low_train_ops = self.get_gan_train_ops(
            gen_scope='model/predict_action',
            dis_scope='model/discriminate_action',
            dis_logit_real=outputs['dis_logit_real_low'],
            dis_logit_gen=outputs['dis_logit_gen_low'],
            gradient_penalty=outputs['gradient_penalty_low'],
            auxiliary_loss=reg_loss * REG_LOSS_WEIGHT,
            scope='gan_low')

        ####
        # Combined train ops.
        ####
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.name_scope('variables'):
            for var in variables:
                tf.summary.histogram(var.name, var)
        eval_ops = self.add_summaries(batch, outputs)
        eval_op = tf.group(*eval_ops)

        train_ops = gan_namedtuples.GANTrainOps(
            (high_train_ops.generator_train_op,
             low_train_ops.generator_train_op,
             dyn_train_op,
             eval_op),
            (high_train_ops.discriminator_train_op,
             low_train_ops.discriminator_train_op),
            training_util.get_or_create_global_step().assign_add(1))
        return train_ops

    def get_gan_train_ops(self,
                          gen_scope,
                          dis_scope,
                          dis_logit_real,
                          dis_logit_gen,
                          gradient_penalty,
                          auxiliary_loss=0.0,
                          scope=None):
        with tf.name_scope(scope):
            gen_lr = tf.train.exponential_decay(
                      learning_rate=0.0001,
                      global_step=tf.train.get_or_create_global_step(),
                      decay_steps=100000,
                      decay_rate=0.9,
                      staircase=True)
            dis_lr = 0.001

            tf.summary.scalar('lr_gen', gen_lr)
            tf.summary.scalar('lr_dis', dis_lr)
            tf.summary.histogram('dis_prob_real', tf.sigmoid(dis_logit_real))
            tf.summary.histogram('dis_prob_gen', tf.sigmoid(dis_logit_gen))
            tf.summary.histogram('gradient_penalty', gradient_penalty)

        with tf.variable_scope(gen_scope) as gen_scope:
            pass
        with tf.variable_scope(dis_scope) as dis_scope:
            pass

        dis_vars = framework.get_trainable_variables(dis_scope)
        gen_vars = framework.get_trainable_variables(gen_scope)

        logger.info('num_gen_vars: %d, num_dis_vars: %d',
                    len(gen_vars), len(dis_vars))
        assert len(gen_vars) > 0
        assert len(dis_vars) > 0

        gen_inputs = None
        gen_data = None
        real_data = None
        gen_fn = None
        dis_fn = None
        gan_model = gan_namedtuples.GANModel(
            gen_inputs, gen_data,
            gen_vars, gen_scope, gen_fn,
            real_data,
            dis_logit_real, dis_logit_gen,
            dis_vars, dis_scope, dis_fn)

        # GAN loss.
        gp_loss = gradient_penalty * GRADIENT_PENALTY_WEIGHT
        gan_loss = tfgan.gan_loss(gan_model, add_summaries=True)
        gan_loss = gan_namedtuples.GANLoss(
            gan_loss.generator_loss + auxiliary_loss,
            gan_loss.discriminator_loss + gp_loss)

        # Gan train ops.
        gen_opt = tf.train.RMSPropOptimizer(gen_lr, decay=.9, momentum=0.1)
        dis_opt = tf.train.RMSPropOptimizer(dis_lr, decay=.95, momentum=0.1)
        train_ops = tfgan.gan_train_ops(
            gan_model,
            gan_loss,
            generator_optimizer=gen_opt,
            discriminator_optimizer=dis_opt,
            check_for_unused_update_ops=False,
            summarize_gradients=True)
        return train_ops

