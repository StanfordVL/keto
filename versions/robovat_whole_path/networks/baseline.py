"""Latent Dynamics Model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf
from tf_agents.utils import nest_utils

from robovat.networks import mpc
from robovat.networks.mpc import expand_and_tile
from robovat.networks.mpc import prune
from robovat.utils.logging import logger  # NOQA

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


VALID_THRESH = 0.9


class VMPC(mpc.MPC):
    """Structured Model Predictive Controller."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(VMPC, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training,
            config=config,
            name=name)

        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

        self.is_training = is_training
        self.dim_z = config.DIM_Z
        self.num_samples = config.NUM_SAMPLES

    def call(self, observation, z):
        """Call the network in the policy."""
        raise NotImplementedError

    def predict(self, observation, z, use_prune):
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            action = []
            pred_state = []
            valid = []
            reward = []
            termination = []
            indices = []

            init_state = self.get_inputs_online(observation)
            state_t = nest.map_structure(
                lambda x: expand_and_tile(x, self.num_samples),
                nest.map_structure(lambda x: x[0], init_state))
            action_t = None
            pred_state_t = None
            valid_t = None
            reward_t = None
            termination_t = None
            indices_t = tf.range(0, self.num_samples, dtype=tf.int64)
            has_terminated = tf.zeros([self.num_samples], dtype=tf.bool)
            for t in range(self.num_steps):
                if t > 0:
                    state_t = nest.map_structure(
                        lambda x: tf.gather(x, indices_t), state_t)
                    has_terminated = tf.gather(has_terminated, indices_t)

                state_t = self.encode_state(state_t)
                action_t = self.predict_action(state_t, z[t])
                pred_state_t, valid_logit_t = self.forward_dynamics(
                    state_t, action_t)
                valid_prob_t = tf.sigmoid(valid_logit_t)
                valid_t = tf.squeeze(tf.greater(valid_prob_t, VALID_THRESH), -1)
                invalid_t = tf.logical_not(valid_t)

                reward_t, termination_t = self.get_reward(state_t, pred_state_t)
                reward_t *= tf.to_float(valid_t)
                reward_t = tf.where(
                    has_terminated,
                    tf.zeros_like(reward_t),
                    reward_t)

                if use_prune:
                    with tf.variable_scope('prune', reuse=tf.AUTO_REUSE):
                        indices_t = prune(
                            has_terminated, termination_t, None, reward_t)
                else:
                    indices_t = tf.range(0, self.num_samples, dtype=tf.int64)

                termination_t = tf.logical_or(termination_t, invalid_t)
                has_terminated = tf.logical_or(has_terminated, termination_t)
                termination_t = has_terminated
                termination.append(termination_t)

                action.append(action_t)
                pred_state.append(pred_state_t)
                valid.append(valid_t)
                reward.append(reward_t)
                indices.append(indices_t)

                # Update.
                state_t = pred_state_t

            # Gather selected plans.
            new_indices = [None] * self.num_steps
            new_indices[self.num_steps - 1] = tf.identity(
                indices[self.num_steps - 1])
            for t in range(self.num_steps - 2, -1, -1):
                new_indices[t] = tf.gather(indices[t], new_indices[t + 1])

            for t in range(self.num_steps):
                action[t] = nest.map_structure(
                    lambda x: tf.gather(x, new_indices[t]), action[t])
                pred_state[t] = nest.map_structure(
                    lambda x: tf.gather(x, new_indices[t]), pred_state[t])
                valid[t] = tf.gather(valid[t], new_indices[t])
                reward[t] = tf.gather(reward[t], new_indices[t])
                termination[t] = tf.gather(termination[t], new_indices[t])

            action = nest_utils.stack_nested_tensors(action)
            pred_state = nest_utils.stack_nested_tensors(pred_state)
            valid = valid[0]
            reward = tf.stack(reward)
            termination = tf.stack(termination)
            indices = new_indices[0]

            return {
                'action': action,
                'pred_state': pred_state,
                'valid': valid, 
                'reward': reward, 
                'termination': termination,
                'indices': indices, 
            }

    def forward(self, batch):
        """Forward the network in the policy or on the dataset."""
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            outputs = dict()

            state, action, next_state = self.get_inputs_from_batch(batch)
            # state: [batch_size, shape_state]
            # action: [batch_size, shape_action]
            # next_state: [batch_size, shape_state]

            state = self.encode_state(state)
            # next_state = self.encode_state(next_state)
            # outputs['encoded_state'] = state
            # outputs['encoded_next_state'] = next_state

            if self.use_point_cloud:
                next_state = self.encode_state(next_state)
                outputs['encoded_state'] = state
                outputs['encoded_next_state'] = next_state

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
            z_mean, z_stddev = self.infer_z(state, action)
            z = self.sample_z(z_mean, z_stddev)
            outputs['z_mean'] = z_mean
            outputs['z_stddev'] = z_stddev
            outputs['z'] = z

            ####
            # Prediction
            ####
            action_decoded = self.predict_action(state, z)
            outputs['action'] = action_decoded

            return outputs

    def sample_z(self, mean, stddev):
        with tf.variable_scope('sample_z'):
            z = mean + stddev * tf.random_normal(
                tf.shape(stddev), 0., 1., dtype=tf.float32)
            return tf.identity(z, name='z')

    def infer_z(self, state, action, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('infer_z', reuse=tf.AUTO_REUSE):
            return self._infer_z(state, action, is_training)

    def predict_action(self, state, z, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_action', reuse=tf.AUTO_REUSE):
            return self._predict_action(state, z, is_training)

    def forward_dynamics(self, state, action, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('forward_dynamics', reuse=tf.AUTO_REUSE):
            return self._forward_dynamics(state, action, is_training)

    @abc.abstractmethod
    def _infer_z(self, state, action, is_training):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_action(self, state, z, is_training):
        raise NotImplementedError

    @abc.abstractmethod
    def _forward_dynamics(self, state, action, is_training):
        raise NotImplementedError


class Sectar(mpc.MPC):
    """Structured Model Predictive Controller."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(Sectar, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training,
            config=config,
            name=name)

        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

        self.is_training = is_training
        self.dim_z = config.DIM_Z
        self.num_samples = config.NUM_SAMPLES

    def call(self, observation, z):
        """Call the network in the policy."""
        raise NotImplementedError

    def predict(self, observation, z, use_prune):
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            action = []
            pred_state = []
            reward = []
            termination = []
            indices = []

            init_state = self.get_inputs_online(observation)
            state_t = nest.map_structure(
                lambda x: expand_and_tile(x, self.num_samples),
                nest.map_structure(lambda x: x[0], init_state))
            action_t = None
            pred_state_t = None
            reward_t = None
            termination_t = None
            indices_t = tf.range(0, self.num_samples, dtype=tf.int64)
            has_terminated = tf.zeros([self.num_samples], dtype=tf.bool)
            for t in range(self.num_steps):
                if t > 0:
                    state_t = nest.map_structure(
                        lambda x: tf.gather(x, indices_t), state_t)
                    has_terminated = tf.gather(has_terminated, indices_t)

                state_t = self.encode_state(state_t)

                action_t = self.predict_action(state_t, z[t])
                pred_state_t = self.predict_transition(state_t, z[t])

                reward_t, termination_t = self.get_reward(state_t, pred_state_t)
                reward_t = tf.where(
                    has_terminated,
                    tf.zeros_like(reward_t),
                    reward_t)

                if use_prune:
                    with tf.variable_scope('prune', reuse=tf.AUTO_REUSE):
                        indices_t = prune(
                            has_terminated, termination_t, None, reward_t)
                else:
                    indices_t = tf.range(0, self.num_samples, dtype=tf.int64)

                has_terminated = tf.logical_or(has_terminated, termination_t)
                termination_t = has_terminated
                termination.append(termination_t)

                action.append(action_t)
                pred_state.append(pred_state_t)
                reward.append(reward_t)
                indices.append(indices_t)

                # Update.
                state_t = pred_state_t

            # Gather selected plans.
            new_indices = [None] * self.num_steps
            new_indices[self.num_steps - 1] = tf.identity(
                indices[self.num_steps - 1])
            for t in range(self.num_steps - 2, -1, -1):
                new_indices[t] = tf.gather(indices[t], new_indices[t + 1])

            for t in range(self.num_steps):
                action[t] = nest.map_structure(
                    lambda x: tf.gather(x, new_indices[t]), action[t])
                pred_state[t] = nest.map_structure(
                    lambda x: tf.gather(x, new_indices[t]), pred_state[t])
                reward[t] = tf.gather(reward[t], new_indices[t])
                termination[t] = tf.gather(termination[t], new_indices[t])

            action = nest_utils.stack_nested_tensors(action)
            pred_state = nest_utils.stack_nested_tensors(pred_state)
            reward = tf.stack(reward)
            termination = tf.stack(termination)
            indices = new_indices[0]

            return {
                'action': action,
                'pred_state': pred_state,
                'reward': reward, 
                'termination': termination,
                'indices': indices, 
            }

    def forward(self, batch):
        """Forward the network in the policy or on the dataset."""
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            outputs = dict()

            state, action, next_state = self.get_inputs_from_batch(batch)
            # state: [batch_size, shape_state]
            # action: [batch_size, shape_action]
            # next_state: [batch_size, shape_state]

            state = self.encode_state(state)
            # next_state = self.encode_state(next_state)
            # outputs['encoded_state'] = state
            # outputs['encoded_next_state'] = next_state

            if self.use_point_cloud:
                next_state = self.encode_state(next_state)
                outputs['encoded_state'] = state
                outputs['encoded_next_state'] = next_state

            ####
            # Inference
            ####
            z_mean, z_stddev = self.infer_z(state, next_state)
            z = self.sample_z(z_mean, z_stddev)
            outputs['z_mean'] = z_mean
            outputs['z_stddev'] = z_stddev
            outputs['z'] = z

            ####
            # Prediction
            ####
            pred_state = self.predict_transition(state, z)
            outputs['pred_state'] = pred_state

            action_decoded = self.predict_action(state, z)
            outputs['action'] = action_decoded

            return outputs

    def sample_z(self, mean, stddev):
        with tf.variable_scope('sample_z'):
            z = mean + stddev * tf.random_normal(
                tf.shape(stddev), 0., 1., dtype=tf.float32)
            return tf.identity(z, name='z')

    def infer_z(self, state, next_state, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('infer_z', reuse=tf.AUTO_REUSE):
            return self._infer_z(state, next_state, is_training)

    def predict_action(self, state, z, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_action', reuse=tf.AUTO_REUSE):
            return self._predict_action(state, z, is_training)

    def predict_transition(self, state, z, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_transition', reuse=tf.AUTO_REUSE):
            return self._predict_transition(state, z, is_training)

    @abc.abstractmethod
    def _infer_z(self, state, next_state, is_training):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_action(self, state, next_state, is_training):
        raise NotImplementedError
