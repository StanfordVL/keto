"""Latent ldmamics Model problems.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np  # NOQA
import tensorflow as tf

from robovat.problems import problem
from robovat.utils import nest_utils
from robovat.utils import loss_utils

slim = tf.contrib.slim
nest = tf.contrib.framework.nest
Reduction = tf.losses.Reduction


class BrownianProblem(problem.Problem):
    """Latent ldmamics Model Problem."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training):
        """Initialize."""
        super(BrownianProblem, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training)

        # Input tensor spec.
        _movable_spec = time_step_spec.observation['movable']
        _movable_id_spec = action_spec['movable_id']
        _motion_spec = action_spec['motion']
        spec_list = [
            tf.TensorSpec(_movable_spec.shape,
                          _movable_spec.dtype,
                          'movable'),
            tf.TensorSpec(_movable_spec.shape,
                          _movable_spec.dtype,
                          'next_movable'),
            tf.TensorSpec(_movable_id_spec.shape,
                          _movable_id_spec.dtype,
                          'movable_id'),
            tf.TensorSpec(_motion_spec.shape,
                          _motion_spec.dtype,
                          'motion'),
        ]
        self._spec = OrderedDict(
            [(spec.name, spec) for spec in spec_list])

    def convert_trajectory(self, trajectory):
        """Convert trajectory."""
        observation = trajectory.observation
        action = trajectory.action
        num_steps = len(trajectory.reward)

        data_list = []
        for i in range(num_steps - 1):
            data = OrderedDict([
                ('movable', observation['movable'][i]),
                ('next_movable', observation['movable'][i + 1]),
                ('movable_id', action['movable_id'][i]),
                ('motion', action['motion'][i]),
            ])

            data_list.append(data)

        return data_list

    def preprocess(self, batch):
        """Augment batched point cloud."""
        with tf.variable_scope('preprocess'):
            state = {'movable': batch['movable']}
            action = {'motion': batch['motion']}
            next_state = {'movable': batch['next_movable']}
            return {
                'state': state,
                'action': action,
                'next_state': next_state,
                'mode_label': batch['movable_id'],
            }

    def normal_kld(self, z, z_mean, z_stddev, weights=1.0):
        return self._normal_kld(z, z_mean, z_stddev, weights)

    def _normal_kld(self, z, z_mean, z_stddev, weights=1.0,
                    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
        kld_array = (loss_utils.log_normal(z, z_mean, z_stddev) -
                     loss_utils.log_normal(z, 0.0, 1.0))
        return tf.losses.compute_weighted_loss(
            kld_array, weights, reduction=reduction)

    def state_loss(self, targets, outputs, weights=1.0):
        return self._state_reconst_loss(targets, outputs, weights)

    def _state_reconst_loss(self, targets, outputs, weights=1.0,
                            reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
        loss_0 = loss_utils.l2_loss(
            targets=targets['movable'][:, 0],
            outputs=outputs['movable'][:, 0],
            weights=weights,
            reduction=reduction)
        loss_1 = loss_utils.l2_loss(
            targets=targets['movable'][:, 1],
            outputs=outputs['movable'][:, 1],
            weights=weights,
            reduction=reduction)
        loss_2 = loss_utils.l2_loss(
            targets=targets['movable'][:, 2],
            outputs=outputs['movable'][:, 2],
            weights=weights,
            reduction=reduction)
        return loss_0 + loss_1 + loss_2

    def loss(self, targets, outputs):
        weights = 1.0
        tf.summary.histogram('valid_weights', weights)

        with tf.variable_scope('loss_computation'):
            z_kld = self.normal_kld(
                outputs['z'],
                outputs['z_mean'],
                outputs['z_stddev'],
                weights)

            state_loss = self.state_loss(
                targets['next_state'],
                outputs['next_state'],
                weights)

        # Summary.
        with tf.variable_scope('loss_breakdown'):
            tf.summary.scalar('z_kld', z_kld)
            tf.summary.scalar('state_loss', state_loss)

        return state_loss * 1e2 + z_kld
    
    def add_summaries(self, targets, outputs):
        """Add summaries."""
        eval_ops = []

        with tf.variable_scope('z_all'):
            tf.summary.histogram('z', outputs['z'])
            tf.summary.histogram('z_mean', outputs['z_mean'])
            tf.summary.histogram('z_stddev', outputs['z_stddev'])

        eval_ops += self.add_state_summaries(targets, outputs)

        return eval_ops

    def add_state_summaries(self, targets, outputs):
        with tf.variable_scope('state_target'):
            next_movable = targets['next_state']['movable']
            tf.summary.histogram('m0_x', next_movable[..., 0, 0])
            tf.summary.histogram('m0_y', next_movable[..., 0, 1])
            tf.summary.histogram('m1_x', next_movable[..., 1, 0])
            tf.summary.histogram('m1_y', next_movable[..., 1, 1])
            tf.summary.histogram('m2_x', next_movable[..., 2, 0])
            tf.summary.histogram('m2_y', next_movable[..., 2, 1])

        with tf.variable_scope('state_output'):
            next_movable = outputs['next_state']['movable']
            tf.summary.histogram('m0_x', next_movable[..., 0, 0])
            tf.summary.histogram('m0_y', next_movable[..., 0, 1])
            tf.summary.histogram('m1_x', next_movable[..., 1, 0])
            tf.summary.histogram('m1_y', next_movable[..., 1, 1])
            tf.summary.histogram('m2_x', next_movable[..., 2, 0])
            tf.summary.histogram('m2_y', next_movable[..., 2, 1])

        # TODO(debug)
        with tf.variable_scope('state_samp'):
            next_movable = outputs['next_state_samp']['movable']
            tf.summary.histogram('m0_x', next_movable[..., 0, 0])
            tf.summary.histogram('m0_y', next_movable[..., 0, 1])
            tf.summary.histogram('m1_x', next_movable[..., 1, 0])
            tf.summary.histogram('m1_y', next_movable[..., 1, 1])
            tf.summary.histogram('m2_x', next_movable[..., 2, 0])
            tf.summary.histogram('m2_y', next_movable[..., 2, 1])

        with tf.variable_scope('d_state_target'):
            curr_movable = targets['state']['movable']
            next_movable = targets['next_state']['movable']
            d_movable = next_movable - curr_movable
            tf.summary.histogram('m0_dx', d_movable[..., 0, 0])
            tf.summary.histogram('m0_dy', d_movable[..., 0, 1])
            tf.summary.histogram('m1_dx', d_movable[..., 1, 0])
            tf.summary.histogram('m1_dy', d_movable[..., 1, 1])
            tf.summary.histogram('m2_dx', d_movable[..., 2, 0])
            tf.summary.histogram('m2_dy', d_movable[..., 2, 1])

        with tf.variable_scope('d_state_output'):
            curr_movable = targets['state']['movable']
            next_movable = outputs['next_state']['movable']
            d_movable = next_movable - curr_movable
            tf.summary.histogram('m0_dx', d_movable[..., 0, 0])
            tf.summary.histogram('m0_dy', d_movable[..., 0, 1])
            tf.summary.histogram('m1_dx', d_movable[..., 1, 0])
            tf.summary.histogram('m1_dy', d_movable[..., 1, 1])
            tf.summary.histogram('m2_dx', d_movable[..., 2, 0])
            tf.summary.histogram('m2_dy', d_movable[..., 2, 1])

        # TODO(debug)
        with tf.variable_scope('d_state_samp'):
            curr_movable = targets['state']['movable']
            next_movable = outputs['next_state_samp']['movable']
            d_movable = next_movable - curr_movable
            tf.summary.histogram('m0_dx', d_movable[..., 0, 0])
            tf.summary.histogram('m0_dy', d_movable[..., 0, 1])
            tf.summary.histogram('m1_dx', d_movable[..., 1, 0])
            tf.summary.histogram('m1_dy', d_movable[..., 1, 1])
            tf.summary.histogram('m2_dx', d_movable[..., 2, 0])
            tf.summary.histogram('m2_dy', d_movable[..., 2, 1])

        return []


class MixtureBrownianProblem(BrownianProblem):
    """Latent ldmamics Model Problem."""

    def normal_kld(self, z, z_mean, z_stddev, mode_posterior, weights=1.0):
        if not isinstance(weights, float):
            weights = tf.expand_dims(weights, -1)
        weights *= mode_posterior

        klds = []
        num_modes = int(mode_posterior.get_shape()[1])
        for k in range(num_modes):
            z_k = z[:, k, :]
            z_mean_k = z_mean[:, k, :]
            z_stddev_k = z_stddev[:, k, :]
            klds_k = self._normal_kld(z_k, z_mean_k, z_stddev_k,
                                      reduction=Reduction.NONE)
            klds.append(klds_k)

        klds = tf.stack(klds, axis=1)
        return tf.losses.compute_weighted_loss(klds, weights)

    def state_loss(self, targets, outputs, mode_posterior, weights=1.0):
        if not isinstance(weights, float):
            weights = tf.expand_dims(weights, -1)
        weights *= mode_posterior

        losses = []
        num_modes = int(mode_posterior.get_shape()[1])
        for k in range(num_modes):
            outputs_k = nest_utils.dict_gather(outputs, k)
            losses_k = self._state_reconst_loss(
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
        batch_size = int(outputs['mode_posterior'].get_shape()[0])
        weights = tf.ones([batch_size], dtype=tf.float32)
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

        with tf.variable_scope('loss_unlabeled'):
            state_loss = self.state_loss(
                targets['next_state'],
                outputs['next_state'],
                outputs['mode_posterior'],
                weights)

            z_kld = self.normal_kld(
                outputs['z'],
                outputs['z_mean'],
                outputs['z_stddev'],
                outputs['mode_posterior'],
                weights)

            mode_kld = loss_utils.kl_divergence(
                tf.stop_gradient(outputs['mode_posterior']),
                # outputs['mode_posterior'],
                outputs['mode_prior'],
                weights)

            mode_cond_entropy, mode_entropy = self.get_entropy(
                outputs['mode_posterior'],
                weights)

            loss_unlabeled = (
                state_loss * 1e3
                + z_kld
                + mode_kld * eta
                - mode_entropy * (1 - eta)
                + mode_cond_entropy
                )

        with tf.variable_scope('loss_unlabeled_breakdown'):
            tf.summary.scalar('z_kld', z_kld)
            tf.summary.scalar('state_loss', state_loss)
            tf.summary.scalar('mode_kld', mode_kld)
            tf.summary.scalar('mode_entropy', mode_entropy)
            tf.summary.scalar('mode_cond_entropy', mode_cond_entropy)
            tf.summary.scalar('loss', loss_unlabeled)

        # with tf.variable_scope('loss_labeled'):
        #     num_modes = int(outputs['mode_prior'].get_shape()[1])
        #     mode_label = self._get_mode_label(targets)
        #     mode_label_onehot = tf.one_hot(mode_label, num_modes)
        #     tf.summary.histogram('mode_label', mode_label)
        #
        #     state_loss = self.state_loss(
        #         targets['next_state'],
        #         outputs['next_state'],
        #         mode_label_onehot,
        #         weights)
        #
        #     z_kld = self.normal_kld(
        #         outputs['z'],
        #         outputs['z_mean'],
        #         outputs['z_stddev'],
        #         mode_label_onehot,
        #         weights)
        #
        #     mode_entropy = tf.losses.softmax_cross_entropy(
        #         mode_label_onehot,
        #         outputs['mode_prior']
        #     )
        #
        #     mode_cond_entropy = tf.losses.softmax_cross_entropy(
        #         mode_label_onehot,
        #         outputs['mode_posterior']
        #     )
        #
        #     loss_labeled = (
        #         state_loss * 1e3
        #         + z_kld
        #         + mode_entropy
        #         + mode_cond_entropy
        #         )
        #
        # with tf.variable_scope('loss_labeled_breakdown'):
        #     tf.summary.scalar('z_kld', z_kld)
        #     tf.summary.scalar('state_loss', state_loss)
        #     tf.summary.scalar('mode_entropy', mode_entropy)
        #     tf.summary.scalar('mode_cond_entropy', mode_cond_entropy)
        #     tf.summary.scalar('loss', loss_labeled)

        return loss_labeled + loss_unlabeled

    def _get_mode_label(self, targets):
        # movable = targets['next_state']['movable']
        # position = targets['next_state']['position']
        # dist = tf.norm(movable - tf.expand_dims(position, -2), axis=-1)
        # mode_label = tf.argmin(dist, axis=-1)
        mode_label = targets['mode_label']
        print('mode_label', mode_label)
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
                tf.summary.histogram('z_%d' % k, outputs['z'][:, k])
        with tf.variable_scope('z_mean'):
            for k in range(num_modes):
                tf.summary.histogram(
                    'z_mean_%d' % k, outputs['z_mean'][:, k])
        with tf.variable_scope('z_stddev'):
            for k in range(num_modes):
                tf.summary.histogram(
                    'z_stddev_%d' % k, outputs['z_stddev'][:, k])

        eval_ops += self.add_state_summaries(targets, outputs)

        return eval_ops

    def add_state_summaries(self, targets, outputs):
        with tf.variable_scope('state_target'):
            next_movable = targets['next_state']['movable']
            tf.summary.histogram('m0_x', next_movable[..., 0, 0])
            tf.summary.histogram('m0_y', next_movable[..., 0, 1])
            tf.summary.histogram('m1_x', next_movable[..., 1, 0])
            tf.summary.histogram('m1_y', next_movable[..., 1, 1])
            tf.summary.histogram('m2_x', next_movable[..., 2, 0])
            tf.summary.histogram('m2_y', next_movable[..., 2, 1])

        num_modes = outputs['mode_posterior'].get_shape()[1]
        for k in range(num_modes):
            with tf.variable_scope('state_output_%d' % k):
                output_state_k = nest_utils.dict_gather(
                    outputs['next_state'], k)
                next_movable = output_state_k['movable']
                tf.summary.histogram('m0_x', next_movable[..., 0, 0])
                tf.summary.histogram('m0_y', next_movable[..., 0, 1])
                tf.summary.histogram('m1_x', next_movable[..., 1, 0])
                tf.summary.histogram('m1_y', next_movable[..., 1, 1])
                tf.summary.histogram('m2_x', next_movable[..., 2, 0])
                tf.summary.histogram('m2_y', next_movable[..., 2, 1])

        # TODO(debug)
        for k in range(num_modes):
            with tf.variable_scope('state_samp_%d' % k):
                output_state_k = nest_utils.dict_gather(
                    outputs['next_state_samp'], k)
                next_movable = output_state_k['movable']
                tf.summary.histogram('m0_x', next_movable[..., 0, 0])
                tf.summary.histogram('m0_y', next_movable[..., 0, 1])
                tf.summary.histogram('m1_x', next_movable[..., 1, 0])
                tf.summary.histogram('m1_y', next_movable[..., 1, 1])
                tf.summary.histogram('m2_x', next_movable[..., 2, 0])
                tf.summary.histogram('m2_y', next_movable[..., 2, 1])

        with tf.variable_scope('d_state_target'):
            curr_movable = targets['state']['movable']
            next_movable = targets['next_state']['movable']
            d_movable = next_movable - curr_movable
            tf.summary.histogram('m0_dx', d_movable[..., 0, 0])
            tf.summary.histogram('m0_dy', d_movable[..., 0, 1])
            tf.summary.histogram('m1_dx', d_movable[..., 1, 0])
            tf.summary.histogram('m1_dy', d_movable[..., 1, 1])
            tf.summary.histogram('m2_dx', d_movable[..., 2, 0])
            tf.summary.histogram('m2_dy', d_movable[..., 2, 1])

        num_modes = outputs['mode_posterior'].get_shape()[1]
        for k in range(num_modes):
            with tf.variable_scope('d_state_output_%d' % k):
                output_state_k = nest_utils.dict_gather(
                    outputs['next_state'], k)
                curr_movable = targets['state']['movable']
                next_movable = output_state_k['movable']
                d_movable = next_movable - curr_movable
                tf.summary.histogram('m0_dx', d_movable[..., 0, 0])
                tf.summary.histogram('m0_dy', d_movable[..., 0, 1])
                tf.summary.histogram('m1_dx', d_movable[..., 1, 0])
                tf.summary.histogram('m1_dy', d_movable[..., 1, 1])
                tf.summary.histogram('m2_dx', d_movable[..., 2, 0])
                tf.summary.histogram('m2_dy', d_movable[..., 2, 1])

        # TODO(debug)
        for k in range(num_modes):
            with tf.variable_scope('d_state_samp_%d' % k):
                output_state_k = nest_utils.dict_gather(
                    outputs['next_state_samp'], k)
                curr_movable = targets['state']['movable']
                next_movable = output_state_k['movable']
                d_movable = next_movable - curr_movable
                tf.summary.histogram('m0_dx', d_movable[..., 0, 0])
                tf.summary.histogram('m0_dy', d_movable[..., 0, 1])
                tf.summary.histogram('m1_dx', d_movable[..., 1, 0])
                tf.summary.histogram('m1_dy', d_movable[..., 1, 1])
                tf.summary.histogram('m2_dx', d_movable[..., 2, 0])
                tf.summary.histogram('m2_dy', d_movable[..., 2, 1])

        return []

