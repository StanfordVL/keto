"""Latent Dynamics Model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf
from tf_agents.networks import network
from tf_agents.utils import nest_utils

from robovat.utils.logging import logger  # NOQA

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


def expand_and_tile(x, multiple, axis=0):
    n_dims = len(x.shape)
    multiples = axis * [1] + [multiple] + (n_dims - axis) * [1]
    return tf.tile(tf.expand_dims(x, axis), multiples)


def interpolate_dict(a, b):
    batch_size = int(list(a.values())[0].shape[0])
    print('batch_size: %d' % batch_size)

    alpha = tf.random_uniform(shape=[batch_size])
    interpolates = dict()
    for key in b.keys():
        difference = b[key] - a[key]
        reshaped_alpha = tf.reshape(
            alpha, [batch_size] + [1] * (difference.shape.ndims - 1))
        interpolates[key] = a[key] + reshaped_alpha * difference

    return interpolates 


def compute_gradient_penalty(discriminate_logit, inputs, target=1.0, eps=1e-10):
    batch_size = int(list(inputs.values())[0].shape[0])
    print('batch_size: %d' % batch_size)

    gradients = tf.gradients(discriminate_logit, list(inputs.values()))

    gradient_squares = 0.0
    for grad in gradients:
        gradient_squares += tf.reduce_sum(
            tf.square(grad), axis=list(range(1, grad.shape.ndims)))

    # Propagate shape information, if possible.
    if isinstance(batch_size, int):
        gradient_squares.set_shape(
            [batch_size] + gradient_squares.shape.as_list()[1:])

    # For numerical stability, add epsilon to the sum before taking the
    # square root. Note tf.norm does not add epsilon.
    slopes = tf.sqrt(gradient_squares + eps)
    penalties = slopes / target - 1.0
    penalties_squared = tf.square(penalties)
    penalty = tf.losses.compute_weighted_loss(
        penalties_squared)

    return penalty


class GMPC(network.Network):
    """Structured Model Predictive Controller."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(GMPC, self).__init__(
            observation_spec=(),
            action_spec=action_spec,
            state_spec=(),
            name=name)

        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

        self.is_training = is_training
        self.config = config

        self.dim_c = config.DIM_C
        self.dim_z = config.DIM_Z

        self.is_training = is_training
        self.config = config
        self.num_steps = config.NUM_STEPS
        self.num_samples = config.NUM_SAMPLES

    def call(self, observation, z, c):
        """Call the network in the policy."""
        raise NotImplementedError

    def high_level_predict(self, observation, c):
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            pred_state = []
            reward = []
            termination = []

            init_state = self.get_inputs_online(observation)
            state_t = nest.map_structure(
                lambda x: expand_and_tile(x, self.num_samples),
                nest.map_structure(lambda x: x[0], init_state))
            pred_state_t = None
            reward_t = None

            is_terminated = tf.zeros([self.num_samples], dtype=tf.bool)
            for t in range(self.num_steps):
                state_t = self.encode_state(state_t)
                pred_state_t = self.predict_transition(state_t, c[t])

                reward_t, termination_t = self.get_reward(state_t, pred_state_t)
                reward_t *= tf.to_float(tf.logical_not(is_terminated))

                is_terminated = tf.logical_or(is_terminated, termination_t)
                termination_t = is_terminated
                termination.append(termination_t)

                pred_state.append(pred_state_t)
                reward.append(reward_t)

                # Update.
                state_t = pred_state_t

            pred_state = nest_utils.stack_nested_tensors(pred_state)
            reward = tf.stack(reward)
            termination = tf.stack(termination)

            return {
                'pred_state': pred_state,
                'reward': reward, 
                'termination': termination,
            }

    def low_level_predict(self, observation, z, c):
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            action = []
            pred_state = []
            valid = []
            reward = []
            termination = []

            init_state = self.get_inputs_online(observation)
            state_t = nest.map_structure(
                lambda x: expand_and_tile(x, self.num_samples),
                nest.map_structure(lambda x: x[0], init_state))
            action_t = None
            pred_state_t = None
            valid_t = None
            reward_t = None

            is_terminated = tf.zeros([self.num_samples], dtype=tf.bool)
            for t in range(self.num_steps):
                state_t = self.encode_state(state_t)
                pred_state_high_t = self.predict_transition(state_t, c[t])
                action_t = self.predict_action(state_t, z[t], pred_state_high_t)
                pred_state_t, valid_logit_t = self.forward_dynamics(
                    state_t, action_t)
                valid_t = tf.squeeze(tf.greater(valid_logit_t, 0.0), -1)

                reward_t, termination_t = self.get_reward(state_t, pred_state_t)
                reward_t *= tf.to_float(valid_t)
                reward_t *= tf.to_float(tf.logical_not(is_terminated))

                invalid_t = tf.logical_not(valid_t)
                termination_t = tf.logical_or(termination_t, invalid_t)
                is_terminated = tf.logical_or(is_terminated, termination_t)
                termination_t = is_terminated
                termination.append(termination_t)

                action.append(action_t)
                pred_state.append(pred_state_t)
                valid.append(valid_t)
                reward.append(reward_t)

                # Update.
                state_t = pred_state_t

            action = nest_utils.stack_nested_tensors(action)
            pred_state = nest_utils.stack_nested_tensors(pred_state)
            valid = valid[0]
            reward = tf.stack(reward)
            termination = tf.stack(termination)

            return {
                'action': action,
                'pred_state': pred_state,
                'valid': valid, 
                'reward': reward, 
                'termination': termination,
            }

    def forward(self, batch):
        """Forward the network in the policy or on the dataset."""
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            outputs = dict()

            state, action, next_state = self.get_inputs_from_batch(batch)
            state = self.encode_state(state)
            next_state = self.encode_state(next_state)
            # state: [batch_size, shape_state]
            # action: [batch_size, shape_action]
            # next_state: [batch_size, shape_state]

            ####
            # Dynamics
            ####
            pred_state, valid_logit = self.forward_dynamics(state, action)
            outputs['pred_state'] = pred_state
            outputs['valid_logit'] = valid_logit
            outputs['valid_prob'] = tf.sigmoid(valid_logit)

            ####
            # Inference
            ####
            batch_size = int(state['position'].shape[0])

            c = self.sample_c(batch_size)
            outputs['c'] = c

            z = self.sample_z(batch_size)
            outputs['z'] = z

            ####
            # Prediction (High-level)
            ####
            pred_state_high = self.predict_transition(state, c)
            outputs['pred_state_high'] = pred_state_high

            dis_logit_real_high = self.discriminate_state(
                state, next_state)
            outputs['dis_logit_real_high'] = dis_logit_real_high

            dis_logit_gen_high = self.discriminate_state(
                state, pred_state_high)
            outputs['dis_logit_gen_high'] = dis_logit_gen_high

            gradient_penalty_high = self.state_gradient_penalty(
                state, next_state, pred_state_high)
            outputs['gradient_penalty_high'] = gradient_penalty_high

            ####
            # Prediction (Low-level)
            ####
            pred_state_high = self.encode_state(pred_state_high)
            pred_state_high = nest.map_structure(
                lambda x: tf.stop_gradient(x), pred_state_high)
            outputs['pred_state_high_sg'] = pred_state_high

            action_gen = self.predict_action(state, z, pred_state_high)
            outputs['action'] = action_gen

            dis_logit_real_low = self.discriminate_action(
                state, pred_state_high, action)
            outputs['dis_logit_real_low'] = dis_logit_real_low

            dis_logit_gen_low = self.discriminate_action(
                state, pred_state_high, action_gen)
            outputs['dis_logit_gen_low'] = dis_logit_gen_low

            gradient_penalty_low = self.action_gradient_penalty(
                state, pred_state_high, action, action_gen)
            outputs['gradient_penalty_low'] = gradient_penalty_low

            ####
            # Regularization
            ####
            pred_state_low, valid_logit_low = self.forward_dynamics(
                state, action_gen)
            outputs['pred_state_low'] = pred_state_low
            outputs['valid_logit_low'] = valid_logit_low

            return outputs

    def get_inputs_from_batch(self, batch):
        with tf.variable_scope('get_inputs_from_batch'):
            state = batch['state']
            action = batch['action']
            next_state = batch['next_state']
            return state, action, next_state

    def state_gradient_penalty(self,
                               state,
                               pred_state_real,
                               pred_state_gen):
        with tf.variable_scope('pred_state_interpolate'):
            interpolates = interpolate_dict(pred_state_real, pred_state_gen)

        dis_logit = self.discriminate_state(state, interpolates,
                                            is_training=True)

        with tf.variable_scope('pred_state_gradient_penalty'):
            penalty = compute_gradient_penalty(dis_logit, interpolates)

        return penalty

    def action_gradient_penalty(self,
                                state,
                                next_state,
                                action_real,
                                action_gen):
        with tf.variable_scope('action_interpolate'):
            interpolates = interpolate_dict(action_real, action_gen)

        dis_logit = self.discriminate_action(state, next_state, interpolates,
                                             is_training=True)

        with tf.variable_scope('action_gradient_penalty'):
            penalty = compute_gradient_penalty(dis_logit, interpolates)

        return penalty

    def sample_c(self, batch_size):
        shape = [batch_size, self.dim_c]
        c = tf.random_normal(shape, 0.0, 1.0, dtype=tf.float32)
        return tf.identity(c, name='c')

    def sample_z(self, batch_size):
        shape = [batch_size, self.dim_z]
        z = tf.random_normal(shape, 0.0, 1.0, dtype=tf.float32)
        return tf.identity(z, name='z')

    def encode_state(self, state, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('encode_state', reuse=tf.AUTO_REUSE):
            return self._encode_state(state, is_training)

    def discriminate_state(self, state, next_state, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('discriminate_state', reuse=tf.AUTO_REUSE):
            return self._discriminate_state(state, next_state, is_training)

    def discriminate_action(self, state, next_state, action, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('discriminate_action', reuse=tf.AUTO_REUSE):
            return self._discriminate_action(
                state, next_state, action, is_training)

    def predict_transition(self, state, c, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_transition', reuse=tf.AUTO_REUSE):
            return self._predict_transition(state, c, is_training)

    def predict_action(self, state, next_state, z, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_action', reuse=tf.AUTO_REUSE):
            return self._predict_action(state, z, next_state, is_training)

    def forward_dynamics(self, state, action, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('forward_dynamics', reuse=tf.AUTO_REUSE):
            return self._forward_dynamics(state, action, is_training)

    @abc.abstractmethod
    def get_inputs_online(self, observation, action):
        raise NotImplementedError

    @abc.abstractmethod
    def get_reward(self, state, next_state):
        raise NotImplementedError

    @abc.abstractmethod
    def _discriminate_state(self, state, next_state, is_training):
        raise NotImplementedError

    @abc.abstractmethod
    def _discriminate_action(self, state, next_state, action, is_training):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_transition(self, state, c, is_training):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_action(self, state, next_state, z, is_training):
        raise NotImplementedError

    @abc.abstractmethod
    def _forward_dynamics(self, state, action, is_training):
        raise NotImplementedError

