"""Latent Dynamics Model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf
from tf_agents.networks import network
from tf_agents.utils import nest_utils

from robovat.envs.reward_fns import manip_reward
from robovat.utils.logging import logger  # NOQA

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


def expand_and_tile(x, multiple, axis=0):
    n_dims = len(x.shape)
    multiples = axis * [1] + [multiple] + (n_dims - axis) * [1]
    return tf.tile(tf.expand_dims(x, axis), multiples)


def sample_masked_data(data, mask, num_samples):
    num_masked = tf.reduce_sum(tf.to_int64(mask))
    masked_data = tf.boolean_mask(data, mask)
    sampled_inds = tf.random.uniform([num_samples], 0, num_masked, tf.int64)
    sampled_data = tf.gather(masked_data, sampled_inds)
    return sampled_data


def prune(has_terminated, termination, invalid, reward, reward_thresh=-1.0):
    batch_size = has_terminated.shape[0]
    assert batch_size == termination.shape[0]
    assert batch_size == reward.shape[0]
    if invalid is not None:
        assert batch_size == invalid.shape[0]

    inds = tf.range(0, batch_size, dtype=tf.int64)

    # Remove invalid plans or bad terminated plans.
    remove = tf.less_equal(reward, reward_thresh)
    if invalid is not None:
        remove = tf.logical_or(remove, invalid)

    remove = tf.logical_and(remove, tf.logical_not(has_terminated))
    keep = tf.logical_not(remove)
    alive = tf.logical_and(keep, tf.logical_not(has_terminated))

    keep_inds = tf.boolean_mask(inds, keep)

    num_remove = tf.reduce_sum(tf.to_int64(remove))
    num_alive = tf.reduce_sum(tf.to_int64(alive))
    substitute_inds = tf.cond(
        tf.greater(num_alive, 0),
        lambda: sample_masked_data(inds, alive, num_remove),
        lambda: tf.zeros([num_remove], dtype=tf.int64)
    )

    ret_inds = tf.concat([keep_inds, substitute_inds], axis=0)
    ret_inds = tf.reshape(ret_inds, [batch_size])

    return ret_inds


class MPC(network.Network):
    """Model predictive controller."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(MPC, self).__init__(
            observation_spec=(),
            action_spec=action_spec,
            state_spec=(),
            name=name)

        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

        self.is_training = is_training
        self.num_steps = config.NUM_STEPS
        self.num_samples = config.NUM_SAMPLES
        self.use_point_cloud = config.USE_POINT_CLOUD

        self.reward_fn = manip_reward.get_reward_fn(config.TASK_NAME)

        assert len(self._action_spec['start'].shape) == 1
        assert len(self._action_spec['motion'].shape) == 1
        self.config = {
            'use_point_cloud': config.USE_POINT_CLOUD,

            'dim_z': config.DIM_Z,
            'dim_c': config.DIM_C,

            'dim_fc_z': config.DIM_FC_Z,
            'dim_fc_c': config.DIM_FC_C,
            'dim_fc_state': config.DIM_FC_STATE,
            'dim_fc_action': config.DIM_FC_ACTION,

            'num_bodies': int(time_step_spec.observation['position'].shape[0]),
            'dim_start': int(action_spec['start'].shape[0]),
            'dim_motion': int(action_spec['motion'].shape[0]),
        }

    def predict(self, observation, action, network_state=()):
        """Call the network in the policy."""
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
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
            termination_t = None
            has_terminated = tf.zeros([self.num_samples], dtype=tf.bool)
            for t in range(self.num_steps):
                action_t = nest.map_structure(lambda x: x[t], action)

                state_t = self.encode_state(state_t)
                pred_state_t, valid_logit_t = self.forward_dynamics(
                    state_t, action_t)

                valid_prob_t = tf.sigmoid(valid_logit_t)
                valid_t = tf.squeeze(tf.greater(valid_prob_t, 0.5), -1)
                invalid_t = tf.logical_not(valid_t)

                reward_t, termination_t = self.get_reward(state_t, pred_state_t)
                reward_t *= tf.to_float(valid_t)
                reward_t = tf.where(
                    has_terminated,
                    tf.zeros_like(reward_t),
                    reward_t)

                termination_t = tf.logical_or(termination_t, invalid_t)
                has_terminated = tf.logical_or(has_terminated, termination_t)
                termination_t = has_terminated
                termination.append(termination_t)

                pred_state.append(pred_state_t)
                valid.append(valid_t)
                reward.append(reward_t)

                # Update.
                state_t = pred_state_t

            pred_state = nest_utils.stack_nested_tensors(pred_state)
            valid = valid[0]
            reward = tf.stack(reward)
            termination = tf.stack(termination)

            return {
                'pred_state': pred_state,
                'valid': valid, 
                'reward': reward, 
                'termination': termination,
            }

        """Call the network in the policy."""
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            outputs = dict()

            state = self.get_inputs_online(observation, action)
            # state: [1, shape_state]
            # action: [num_samples, shape_action] or
            #         [num_samples, num_steps, shape_action]

            ####
            # Encode state.
            ####
            state = self.encode_state(state)
            assert state.get_shape()[0] == 1
            state = tf.tile(state, [self.num_samples, 1], name='tiled_h')

            ####
            # Prediction
            ####
            if self.num_steps is None:
                num_steps = 1
            else:
                num_steps = self.num_steps

            pred_state = []
            reward = []

            state_t = state  # TODO
            h_t = state

            for t in range(num_steps):
                if isinstance(action, dict):
                    action_t = nest.map_structure(lambda x: x[:, t], action)
                else:
                    action_t = action[:, t]

                pred_h_t = self.forward_dynamics(h_t, action_t)
                pred_state_t = self.decode_state(pred_h_t)
                reward_t = self.get_reward(state_t, pred_state_t)

                pred_state.append(pred_state_t)
                reward.append(reward_t)

                state_t = pred_state_t
                h_t = pred_h_t

            if self.num_steps is None:
                pred_state = pred_state[0]
                reward = reward[0]
            else:
                pred_state = nest_utils.stack_nested_tensors(pred_state)
                reward = tf.stack(reward)

            outputs['pred_state'] = pred_state
            outputs['reward'] = reward

            return outputs

    def forward(self, batch):
        """Forward the network in the policy or on the dataset."""
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            outputs = dict()

            state, action, next_state = self.get_inputs_from_batch(batch)
            # state: [batch_size, shape_state]
            # action: [batch_size, shape_action]
            # next_state: [batch_size, shape_state]

            state = self.encode_state(state)

            if self.use_point_cloud:
                next_state = self.encode_state(next_state)
                outputs['encoded_state'] = state
                outputs['encoded_next_state'] = next_state

            ####
            # Prediction
            ####
            pred_state, valid_logit = self.forward_dynamics(state, action)
            outputs['pred_state'] = pred_state
            outputs['valid_logit'] = valid_logit
            outputs['valid_prob'] = tf.sigmoid(valid_logit)

            return outputs

    def encode_state(self, state, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('encode_state', reuse=tf.AUTO_REUSE):
            return self._encode_state(state, is_training)

    def forward_dynamics(self, state, action, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('forward_dynamics', reuse=tf.AUTO_REUSE):
            return self._forward_dynamics(state, action, is_training)

    def get_inputs_from_batch(self, batch):
        with tf.variable_scope('get_inputs_from_batch'):
            state = batch['state']
            action = batch['action']
            next_state = batch['next_state']
            return state, action, next_state

    # @abc.abstractmethod
    # def get_inputs_online(self, observation, action):
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def get_reward(self, state, next_state):
    #     raise NotImplementedError

    def get_inputs_online(self, observation):
        with tf.variable_scope('get_inputs_online'):
            state = {
                'position': observation['position'][..., :2],
            }
            if self.use_point_cloud:
                state['point_cloud'] = observation['point_cloud']
            return state

    def get_reward(self, state, next_state):
        if isinstance(state, dict):
            state = state['position'][..., :2]
            next_state = next_state['position'][..., :2]

        reward, termination = tf.py_func(
            self.reward_fn,
            [state, next_state],
            [tf.float32, tf.bool])

        reward = tf.reshape(reward, [self.num_samples])
        termination = tf.reshape(termination, [self.num_samples])
        return reward, termination

    @abc.abstractmethod
    def _encode_state(self, state, action, is_training):
        raise NotImplementedError

    def _forward_dynamics(self, state, action, is_training):
        raise NotImplementedError


class SMPC(MPC):
    """Structured Model Predictive Controller."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(SMPC, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training,
            config=config,
            name=name)

        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

        self.is_training = is_training

        self.dim_c = config.DIM_C
        self.dim_z = config.DIM_Z

        self.num_samples = config.NUM_SAMPLES

    def call(self, observation, z, c):
        """Call the network in the policy."""
        raise NotImplementedError

    def high_level_predict(self, observation, c, use_prune):
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            pred_state = []
            reward = []
            termination = []
            indices = []

            init_state = self.get_inputs_online(observation)
            state_t = nest.map_structure(
                lambda x: expand_and_tile(x, self.num_samples),
                nest.map_structure(lambda x: x[0], init_state))
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
                pred_state_t = self.predict_transition(state_t, c[t])

                reward_t, termination_t = self.get_reward(
                    state_t, pred_state_t)
                reward_t = tf.where(
                    has_terminated,
                    tf.zeros_like(reward_t),
                    reward_t)

                if use_prune:
                    with tf.variable_scope('prune_high', reuse=tf.AUTO_REUSE):
                        indices_t = prune(
                                has_terminated, termination_t, None, reward_t)
                else:
                    indices_t = tf.range(0, self.num_samples, dtype=tf.int64)

                has_terminated = tf.logical_or(
                    has_terminated, termination_t)
                termination_t = has_terminated
                termination.append(termination_t)

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
                pred_state[t] = nest.map_structure(
                    lambda x: tf.gather(x, new_indices[t]), pred_state[t])
                reward[t] = tf.gather(reward[t], new_indices[t])
                termination[t] = tf.gather(termination[t], new_indices[t])

            pred_state = nest_utils.stack_nested_tensors(pred_state)
            reward = tf.stack(reward)
            termination = tf.stack(termination)
            indices = new_indices[0]

            return {
                'pred_state': pred_state,
                'reward': reward, 
                'termination': termination,
                'indices': indices, 
            }

    def low_level_predict(self, observation, z, c, use_prune):
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            init_state = self.get_inputs_online(observation)
            state_t = nest.map_structure(
                lambda x: expand_and_tile(x, self.num_samples),
                nest.map_structure(lambda x: x[0], init_state))

            z_t = z[0]
            c_t = c[0]

            state_t = self.encode_state(state_t)
            high_pred_state_t = self.predict_transition(state_t, c_t)

            action_t = self.predict_action(state_t, z_t, c_t)
            pred_state_t, valid_logit_t = self.forward_dynamics(
                state_t, action_t)
            valid_prob_t = tf.sigmoid(valid_logit_t)
            valid_t = tf.squeeze(tf.greater(valid_prob_t, 0.9), -1)

            dist_t = self.get_state_dist(high_pred_state_t, pred_state_t)
            reward_t = tf.exp(-dist_t)
            reward_t *= tf.to_float(valid_t)

            action_t = nest.map_structure(
                lambda x: tf.expand_dims(x, 0), action_t)
            pred_state_t = nest.map_structure(
                lambda x: tf.expand_dims(x, 0), pred_state_t)
            valid_t = valid_t
            reward_t = tf.expand_dims(reward_t, 0)

            return {
                'action': action_t,
                'pred_state': pred_state_t,
                'valid': valid_t, 
                'reward': reward_t
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
            next_state = self.encode_state(next_state)
            # TODO(Debug)
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
            c_mean, c_stddev = self.infer_c(state, next_state)
            c = self.sample_c(c_mean, c_stddev)
            outputs['c_mean'] = c_mean
            outputs['c_stddev'] = c_stddev
            outputs['c'] = c

            z_mean, z_stddev = self.infer_z(state, action, c)
            z = self.sample_z(z_mean, z_stddev)
            outputs['z_mean'] = z_mean
            outputs['z_stddev'] = z_stddev
            outputs['z'] = z

            ####
            # Prediction (High-level)
            ####
            pred_state_high = self.predict_transition(state, c)
            outputs['pred_state_high'] = pred_state_high

            pred_state_high = self.encode_state(pred_state_high)
            pred_state_high = nest.map_structure(
                lambda x: tf.stop_gradient(x), pred_state_high)
            outputs['pred_state_high_sg'] = pred_state_high

            ####
            # Prediction (Low-level)
            ####
            action_low = self.predict_action(state, z, c)
            outputs['action'] = action_low

            pred_state_low, valid_logit_low = self.forward_dynamics(
                state, action_low)
            outputs['pred_state_low'] = pred_state_low
            outputs['valid_logit_low'] = valid_logit_low

            ####
            # Regularization
            ####
            c = tf.stop_gradient(c)
            z_gen = self.sample_z(tf.zeros_like(z_mean),
                                  tf.zeros_like(z_stddev))
            action_gen = self.predict_action(state, z_gen, c)

            pred_state_gen, valid_logit_gen = self.forward_dynamics(
                state, action_gen)
            outputs['pred_state_gen'] = pred_state_gen
            outputs['valid_logit_gen'] = valid_logit_gen

            c_gen_mean, c_gen_stddev = self.infer_c(state, pred_state_gen)
            c_gen = self.sample_c(c_gen_mean, c_gen_stddev)
            outputs['c_gen_mean'] = c_gen_mean
            outputs['c_gen_stddev'] = c_gen_stddev
            outputs['c_gen'] = c_gen

            return outputs

    def sample_c(self, mean, stddev):
        with tf.variable_scope('sample_c'):
            c = mean + stddev * tf.random_normal(
                tf.shape(stddev), 0., 1., dtype=tf.float32)
            return tf.identity(c, name='c')

    def sample_z(self, mean, stddev):
        with tf.variable_scope('sample_z'):
            z = mean + stddev * tf.random_normal(
                tf.shape(stddev), 0., 1., dtype=tf.float32)
            return tf.identity(z, name='z')

    def infer_c(self, state, next_state, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('infer_c', reuse=tf.AUTO_REUSE):
            return self._infer_c(state, next_state, is_training)

    def infer_z(self, state, action, c, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('infer_z', reuse=tf.AUTO_REUSE):
            return self._infer_z(state, action, c, is_training)

    def predict_transition(self, state, c, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_transition', reuse=tf.AUTO_REUSE):
            return self._predict_transition(state, c, is_training)

    def predict_action(self, state, z, c, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_action', reuse=tf.AUTO_REUSE):
            return self._predict_action(state, z, c, is_training)

    @abc.abstractmethod
    def _infer_c(self, state, next_state, is_training):
        raise NotImplementedError

    @abc.abstractmethod
    def _infer_z(self, state, action, c, is_training):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_transition(self, state, c, is_training):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_action(self, state, z, c, is_training):
        raise NotImplementedError
