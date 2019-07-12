"""Latent ldmamics Model problems.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.gan.python import namedtuples as gan_namedtuples

from robovat.problems import problem
from robovat.utils import loss_utils
from robovat.utils.logging import logger  # NOQA

slim = tf.contrib.slim
nest = tf.contrib.framework.nest
framework = tf.contrib.framework
tfgan = tf.contrib.gan
Reduction = tf.losses.Reduction


USE_GAN = 0
ACTION_RECONS_LOSS = 1e3


class CircleProblem(problem.Problem):
    """Latent ldmamics Model Problem."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training):
        """Initialize."""
        super(CircleProblem, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training)

        # Input tensor spec.
        _movable_spec = time_step_spec.observation['movable']
        _position_spec = action_spec['position']
        spec_list = [
            tf.TensorSpec([1], tf.int64, 'is_effective'),
            tf.TensorSpec(_movable_spec.shape,
                          _movable_spec.dtype,
                          'movable'),
            tf.TensorSpec(_position_spec.shape,
                          _position_spec.dtype,
                          'position'),
        ]
        self._spec = OrderedDict(
            [(spec.name, spec) for spec in spec_list])

    def convert_trajectory(self, trajectory):
        """Convert trajectory."""
        observation = trajectory.observation
        action = trajectory.action
        reward = trajectory.reward
        num_steps = len(trajectory.reward)

        data_list = []
        for i in range(num_steps - 1):
            if reward[i] > 0:
                data = OrderedDict([
                    ('is_effective', np.array([reward[i]], dtype=np.int64)),
                    ('movable', observation['movable'][i]),
                    ('position', action['position'][i]),
                ])

                data_list.append(data)

        return data_list

    def preprocess(self, batch):
        """Augment batched point cloud."""
        with tf.variable_scope('preprocess'):
            state = {'movable': batch['movable']}
            action = {'position': batch['position']}
            return {
                'state': state,
                'action': action,
                'is_effective': batch['is_effective'],
            }

    def get_valid_weights(self, targets):
        return tf.cast(
            tf.squeeze(targets['is_effective'], -1),
            tf.float32)

    def normal_kld(self, z, z_mean, z_stddev, weights=1.0):
        return self._normal_kld(z, z_mean, z_stddev, weights)

    def _normal_kld(self, z, z_mean, z_stddev, weights=1.0,
                    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
        kld_array = (loss_utils.log_normal(z, z_mean, z_stddev) -
                     loss_utils.log_normal(z, 0.0, 1.0))
        return tf.losses.compute_weighted_loss(
            kld_array, weights, reduction=reduction)

    def action_loss(self, targets, outputs, weights=1.0):
        return self._action_reconst_loss(targets, outputs, weights)

    def _action_reconst_loss(self, targets, outputs, weights=1.0,
                             reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
        return loss_utils.l2_loss(
            targets=targets['position'],
            outputs=outputs['position'],
            weights=weights,
            reduction=reduction)

    def loss(self, targets, outputs):
        weights = self.get_valid_weights(targets)
        tf.summary.histogram('valid_weights', weights)

        with tf.variable_scope('loss_computation'):
            z_kld = self.normal_kld(
                outputs['z'],
                outputs['z_mean'],
                outputs['z_stddev'],
                weights)

            action_loss = self.action_loss(
                targets['action'],
                outputs['action'],
                weights)

        # Summary.
        with tf.variable_scope('loss_breakdown'):
            tf.summary.scalar('z_kld', z_kld)
            tf.summary.scalar('action_loss', action_loss)

        return action_loss * ACTION_RECONS_LOSS + z_kld
    
    def add_summaries(self, targets, outputs):
        """Add summaries."""
        eval_ops = []

        with tf.variable_scope('z_all'):
            tf.summary.histogram('z', outputs['z'])
            tf.summary.histogram('z_mean', outputs['z_mean'])
            tf.summary.histogram('z_stddev', outputs['z_stddev'])

        eval_ops += self.add_action_summaries(targets, outputs)

        return eval_ops

    def add_action_summaries(self, targets, outputs):
        with tf.variable_scope('action_target'):
            target_position = targets['action']['position']
            tf.summary.histogram('x', target_position[..., 0])
            tf.summary.histogram('y', target_position[..., 1])

        with tf.variable_scope('action_output'):
            output_position = outputs['action']['position']
            tf.summary.histogram('x', output_position[..., 0])
            tf.summary.histogram('y', output_position[..., 1])

        # TODO(debug)
        with tf.variable_scope('action_pred'):
            output_action = outputs['action_gen']
            output_position = output_action['position']
            tf.summary.histogram('x', output_position[..., 0])
            tf.summary.histogram('y', output_position[..., 1])

        return []


class MixtureCircleProblem(CircleProblem):
    """Latent ldmamics Model Problem."""

    def normal_kld(self, z, z_mean, z_stddev, mode_posterior, weights=1.0):
        if not isinstance(weights, float):
            weights = tf.expand_dims(weights, -1)
        weights *= mode_posterior

        klds = []
        num_modes = int(mode_posterior.get_shape()[1])
        for k in range(num_modes):
            z_k = z[k]
            z_mean_k = z_mean[k]
            z_stddev_k = z_stddev[k]
            klds_k = self._normal_kld(z_k, z_mean_k, z_stddev_k,
                                      reduction=Reduction.NONE)
            klds.append(klds_k)

        klds = tf.stack(klds, axis=1)
        return tf.losses.compute_weighted_loss(klds, weights)

    def action_loss(self, targets, outputs, mode_posterior, weights=1.0):
        if not isinstance(weights, float):
            weights = tf.expand_dims(weights, -1)
        weights *= mode_posterior

        losses = []
        num_modes = int(mode_posterior.get_shape()[1])
        for k in range(num_modes):
            outputs_k = nest.map_structure(lambda x: x[k], outputs)
            losses_k = self._action_reconst_loss(
                targets, outputs_k, reduction=Reduction.NONE)
            losses.append(losses_k)

        losses = tf.stack(losses, axis=1)
        return tf.losses.compute_weighted_loss(losses, weights)

    def get_entropy(self, prob, weights):
        prob_weights = tf.expand_dims(weights, 1) * tf.ones_like(prob)
        marginal_prob = tf.div_no_nan(
            tf.reduce_sum(prob * prob_weights, axis=-2),
            tf.reduce_sum(prob_weights, axis=-2))

        cond_entropy = loss_utils.entropy(prob, weights)
        entropy = loss_utils.entropy(
            marginal_prob, tf.reduce_mean(weights, axis=-1))

        return cond_entropy, entropy

    def loss(self, targets, outputs):
        weights = self.get_valid_weights(targets)
        tf.summary.histogram('valid_weights', weights)

        global_step = slim.get_or_create_global_step()
        boundaries = [10000]
        values = [0.01, 1.0 - 0.01]
        eta = tf.train.piecewise_constant(
            global_step, boundaries, values)
        tf.summary.scalar('global_step_user', global_step)
        tf.summary.scalar('eta', eta)

        # TODO(debug)
        loss_unlabeled = 0.0
        loss_labeled = 0.0

        # with tf.variable_scope('loss_unlabeled'):
        #     action_loss = self.action_loss(
        #         targets['action'],
        #         outputs['action'],
        #         outputs['mode_posterior'],
        #         weights)
        #
        #     z_kld = self.normal_kld(
        #         outputs['z'],
        #         outputs['z_mean'],
        #         outputs['z_stddev'],
        #         outputs['mode_posterior'],
        #         weights)
        #
        #     mode_kld = loss_utils.kl_divergence(
        #         tf.stop_gradient(outputs['mode_posterior']),
        #         # outputs['mode_posterior'],
        #         outputs['mode_prior'],
        #         weights)
        #
        #     mode_cond_entropy, mode_entropy = self.get_entropy(
        #         outputs['mode_posterior'],
        #         weights)
        #
        #     loss_unlabeled = (
        #         action_loss * ACTION_RECONS_LOSS
        #         + z_kld
        #         + mode_kld * eta
        #         - mode_entropy * (1 - eta)
        #         + mode_cond_entropy
        #         )
        #
        # with tf.variable_scope('loss_unlabeled_breakdown'):
        #     tf.summary.scalar('z_kld', z_kld)
        #     tf.summary.scalar('action_loss', action_loss)
        #     tf.summary.scalar('mode_kld', mode_kld)
        #     tf.summary.scalar('mode_entropy', mode_entropy)
        #     tf.summary.scalar('mode_cond_entropy', mode_cond_entropy)
        #     tf.summary.scalar('loss', loss_unlabeled)

        with tf.variable_scope('loss_labeled'):
            num_modes = int(outputs['mode_prior'].get_shape()[1])
            mode_label = self._get_mode_label(targets)
            mode_label_onehot = tf.one_hot(mode_label, num_modes)
            tf.summary.histogram('mode_label', mode_label)

            action_loss = self.action_loss(
                targets['action'],
                outputs['action'],
                mode_label_onehot,
                weights)

            z_kld = self.normal_kld(
                outputs['z'],
                outputs['z_mean'],
                outputs['z_stddev'],
                mode_label_onehot,
                weights)

            mode_entropy = tf.losses.softmax_cross_entropy(
                mode_label_onehot,
                outputs['mode_prior']
            )

            mode_cond_entropy = tf.losses.softmax_cross_entropy(
                mode_label_onehot,
                outputs['mode_posterior']
            )

            loss_labeled = (
                action_loss * ACTION_RECONS_LOSS
                + z_kld
                + mode_entropy
                + mode_cond_entropy
                )

        with tf.variable_scope('loss_labeled_breakdown'):
            tf.summary.scalar('z_kld', z_kld)
            tf.summary.scalar('action_loss', action_loss)
            tf.summary.scalar('mode_entropy', mode_entropy)
            tf.summary.scalar('mode_cond_entropy', mode_cond_entropy)
            tf.summary.scalar('loss', loss_labeled)

        return loss_labeled + loss_unlabeled

    def _get_mode_label(self, targets):
        movable = targets['state']['movable']
        position = targets['action']['position']
        dist = tf.norm(movable - tf.expand_dims(position, -2), axis=-1)
        mode_label = tf.argmin(dist, axis=-1)
        return mode_label
    
    def add_summaries(self, targets, outputs):
        """Add summaries."""
        eval_ops = []
        num_modes = int(outputs['mode_prior'].get_shape()[1])

        with tf.variable_scope('mode_posterior'):
            prob = outputs['mode_posterior']
            tf.summary.histogram(
                'prob_sum', tf.reduce_sum(prob, -1))
            for k in range(num_modes):
                tf.summary.histogram(
                    'prob_%d' % (k), prob[..., k])

        with tf.variable_scope('mode_prior'):
            prob = outputs['mode_prior']
            tf.summary.histogram(
                'prob_sum', tf.reduce_sum(prob, -1))
            for k in range(num_modes):
                tf.summary.histogram(
                    'prob_%d' % (k), prob[..., k])

        with tf.variable_scope('mode_prior_marginal'):
            prob = tf.reduce_mean(outputs['mode_posterior'], axis=0)
            tf.summary.histogram(
                'prob_sum', tf.reduce_sum(prob, -1))
            for k in range(num_modes):
                tf.summary.histogram(
                    'prob_%d' % (k), prob[..., k])

        with tf.variable_scope('z_all'):
            tf.summary.histogram('z', outputs['z'])
            tf.summary.histogram('z_mean', outputs['z_mean'])
            tf.summary.histogram('z_stddev', outputs['z_stddev'])
        with tf.variable_scope('z'):
            for k in range(num_modes):
                tf.summary.histogram('z_%d' % k, outputs['z'][k])
        with tf.variable_scope('z_mean'):
            for k in range(num_modes):
                tf.summary.histogram(
                    'z_mean_%d' % k, outputs['z_mean'][k])
        with tf.variable_scope('z_stddev'):
            for k in range(num_modes):
                tf.summary.histogram(
                    'z_stddev_%d' % k, outputs['z_stddev'][k])

        eval_ops += self.add_action_summaries(targets, outputs)

        return eval_ops

    def add_action_summaries(self, targets, outputs):
        with tf.variable_scope('action_target'):
            target_position = targets['action']['position']
            tf.summary.histogram('x', target_position[..., 0])
            tf.summary.histogram('y', target_position[..., 1])

        num_modes = outputs['mode_posterior'].get_shape()[1]
        for k in range(num_modes):
            with tf.variable_scope('action_output_%d' % k):
                output_action_k = nest.map_structure(
                    lambda x: x[k], outputs['action'])
                output_position = output_action_k['position']
                tf.summary.histogram('x', output_position[..., 0])
                tf.summary.histogram('y', output_position[..., 1])

        for k in range(num_modes):
            with tf.variable_scope('action_gen_%d' % k):
                output_action_k = nest.map_structure(
                    lambda x: x[k], outputs['action_gen'])
                output_position = output_action_k['position']
                tf.summary.histogram('x', output_position[..., 0])
                tf.summary.histogram('y', output_position[..., 1])

        return []

    def get_train_op(self,
                     network,
                     batch,
                     learning_rate,
                     clip_gradient_norm=-1):
        if USE_GAN:
            return self.get_gan_train_ops(
                network=network,
                batch=batch)
        else:
            return super(MixtureCircleProblem, self).get_train_op(
                network=network,
                batch=batch,
                learning_rate=learning_rate,
                clip_gradient_norm=clip_gradient_norm)

    def get_gan_train_ops(self, network, batch, gradient_penalty_weight=1e4):
        ####
        # Normal Train Op
        ####
        logger.info('Forwarding network...')
        outputs = network.forward(batch)

        logger.info('Building the loss function...')
        main_loss = self.loss(batch, outputs)

        logger.info('Adding summaries...')
        tf.summary.scalar('loss_main', main_loss)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.name_scope('variables'):
            for var in variables:
                tf.summary.histogram(var.name, var)
        eval_ops = self.add_summaries(batch, outputs)

        logger.info('Building the train_op...')
        slim.get_or_create_global_step()
        learning_rate = 0.0001
        clip_gradient_norm = -1
        optimizer = tf.train.AdamOptimizer(learning_rate)
        learning_rate = self.get_learning_rate(learning_rate)
        train_op = slim.learning.create_train_op(
                main_loss,
                optimizer,
                clip_gradient_norm=clip_gradient_norm,
                summarize_gradients=True)
        train_op = tf.group(train_op, *eval_ops)

        ####
        # GAN Train Op
        ####
        logger.info('Building GAN...')
        with tf.variable_scope('model/action_decoder') as gen_scope:
            pass
        with tf.variable_scope('model/action_discriminator') as dis_scope:
            pass
        discriminator_variables = framework.get_trainable_variables(
            dis_scope)
        generator_variables = framework.get_trainable_variables(
            gen_scope)

        assert len(generator_variables) > 0
        assert len(discriminator_variables) > 0
        print('#gen_vars: %d, #dis_vars: %d' %
              (len(generator_variables), len(discriminator_variables)))

        generator_inputs = None
        generated_data = None
        real_data = None
        generator_fn = None
        discriminator_fn = None

        discriminator_real_outputs = outputs['dis_logit_real']
        discriminator_gen_outputs = outputs['dis_logit_gen']
        with tf.name_scope('gan_discriminator'):
            tf.summary.histogram('dis_real',
                                 tf.sigmoid(outputs['dis_logit_real']))
            tf.summary.histogram('dis_gen',
                                 tf.sigmoid(outputs['dis_logit_gen']))

        gan_model = gan_namedtuples.GANModel(
            generator_inputs, generated_data,
            generator_variables, gen_scope, generator_fn,
            real_data, discriminator_real_outputs,
            discriminator_gen_outputs,
            discriminator_variables, dis_scope,
            discriminator_fn)

        gan_loss = tfgan.gan_loss(
            gan_model,
            add_summaries=True)
        gp_loss = outputs['gradient_penalty'] * gradient_penalty_weight
        gan_loss = gan_namedtuples.GANLoss(
            gan_loss.generator_loss + main_loss,
            gan_loss.discriminator_loss + gp_loss)
        tf.summary.scalar('loss_gradient_penalty', gp_loss)

        gen_lr = tf.train.exponential_decay(
                  learning_rate=0.0001,
                  global_step=tf.train.get_or_create_global_step(),
                  decay_steps=100000,
                  decay_rate=0.9,
                  staircase=True)
        dis_lr = 0.001
        gen_opt = tf.train.RMSPropOptimizer(gen_lr, decay=.9, momentum=0.1)
        dis_opt = tf.train.RMSPropOptimizer(dis_lr, decay=.95, momentum=0.1)
        train_ops = tfgan.gan_train_ops(
            gan_model,
            gan_loss,
            generator_optimizer=gen_opt,
            discriminator_optimizer=dis_opt,
            check_for_unused_update_ops=False,
            summarize_gradients=True)
        tf.summary.scalar('lr_generator', gen_lr)
        tf.summary.scalar('lr_discriminator', dis_lr)

        train_ops = gan_namedtuples.GANTrainOps(
            tf.group(train_op, train_ops.generator_train_op),
            train_ops.discriminator_train_op,
            train_ops.global_step_inc_op,
            train_ops.train_hooks)

        return train_ops
