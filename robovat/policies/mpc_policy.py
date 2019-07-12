"""Multi-stage policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.policies import policy_step
from tf_agents.policies import tf_policy

from robovat import networks
from robovat.policies import cem  # NOQA
from robovat.policies import samplers  # NOQA
from robovat.policies import heuristics  # NOQA


nest = tf.contrib.framework.nest


GAMMA = 0.99
# GAMMA = 1.0


class MPCPolicy(tf_policy.Base):
    """CEM policy for multi-stage tasks."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None):
        self._debug = config.DEBUG

        self._q_network_ctor = getattr(networks, 'PushMPC')

        self._num_samples = config.NUM_SAMPLES
        self._num_steps = config.NUM_STEPS

        self._sampler = samplers.ActionSampler(
            time_step_spec, action_spec, config)

        q_network = tf.make_template(
            'model',
            self._q_network_ctor,
            create_scope_now_=True,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=False,
            config=config)
        self._q_network = q_network()

        super(MPCPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec)

    def _variables(self):
        return self._q_network.variables

    def _action(self, time_step, policy_state, seed):
        action_samples = self._sampler(time_step, self._num_samples, seed)

        observation = time_step.observation
        outputs = self._q_network.predict(observation, action_samples)
        rewards_t = outputs['reward']
        decay = tf.cumprod(GAMMA * tf.ones_like(rewards_t), axis=0) / GAMMA
        rewards = tf.reduce_sum(rewards_t * decay, axis=0)

        # Select final elites.
        _, inds = tf.nn.top_k(rewards, k=1)
        action = nest.map_structure(lambda x: tf.gather(x, inds, axis=1),
                                    action_samples)
        action = nest.map_structure(lambda x: x[0], action)

        return policy_step.PolicyStep(action, policy_state)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')


class VMPCPolicy(tf_policy.Base):
    """CEM policy for multi-stage tasks."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None):
        self._debug = config.DEBUG

        self._q_network_ctor = getattr(networks, 'PushVMPC')

        self._num_samples = config.NUM_SAMPLES
        self._num_steps = config.NUM_STEPS

        self._z_sampler = samplers.ZSampler(
            time_step_spec, action_spec, config)

        q_network = tf.make_template(
            'model',
            self._q_network_ctor,
            create_scope_now_=True,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=False,
            config=config)
        self._q_network = q_network()

        super(VMPCPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec)

    def _variables(self):
        return self._q_network.variables

    def _action(self, time_step, policy_state, seed):
        z_samples = self._z_sampler(time_step, self._num_samples, seed)

        observation = time_step.observation

        outputs = self._q_network.predict(
            observation, z_samples, use_prune=True)
        actions = outputs['action']
        rewards_t = outputs['reward']
        decay = tf.cumprod(GAMMA * tf.ones_like(rewards_t), axis=0) / GAMMA
        rewards = tf.reduce_sum(rewards_t * decay, axis=0)

        # Select final elites.
        _, inds = tf.nn.top_k(rewards, k=1)
        action = nest.map_structure(lambda x: tf.gather(x, inds, axis=1),
                                    actions)
        action = nest.map_structure(lambda x: x[0], action)

        return policy_step.PolicyStep(action, policy_state)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')


class SectarPolicy(tf_policy.Base):
    """CEM policy for multi-stage tasks."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None):
        self._debug = config.DEBUG

        self._q_network_ctor = getattr(networks, 'PushSectar')

        self._num_samples = config.NUM_SAMPLES
        self._num_steps = config.NUM_STEPS

        self._z_sampler = samplers.ZSampler(
            time_step_spec, action_spec, config)

        q_network = tf.make_template(
            'model',
            self._q_network_ctor,
            create_scope_now_=True,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=False,
            config=config)
        self._q_network = q_network()

        super(SectarPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec)

    def _variables(self):
        return self._q_network.variables

    def _action(self, time_step, policy_state, seed):
        z_samples = self._z_sampler(time_step, self._num_samples, seed)

        observation = time_step.observation

        outputs = self._q_network.predict(
            observation, z_samples, use_prune=True)
        actions = outputs['action']
        rewards_t = outputs['reward']
        decay = tf.cumprod(GAMMA * tf.ones_like(rewards_t), axis=0) / GAMMA
        rewards = tf.reduce_sum(rewards_t * decay, axis=0)

        # Select final elites.
        _, inds = tf.nn.top_k(rewards, k=1)
        action = nest.map_structure(lambda x: tf.gather(x, inds, axis=1),
                                    actions)
        action = nest.map_structure(lambda x: x[0], action)

        return policy_step.PolicyStep(action, policy_state)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')


class SMPCPolicy(tf_policy.Base):
    """CEM policy for multi-stage tasks."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None):
        self._debug = config.DEBUG

        self._q_network_ctor = getattr(networks, config.NETWORK)

        self._num_steps = config.NUM_STEPS
        self._num_samples = config.NUM_SAMPLES
        self._num_c_elites = config.NUM_C_ELITES
        assert self._num_samples % self._num_c_elites == 0

        self._z_sampler = samplers.ZSampler(
            time_step_spec, action_spec, config)
        self._c_sampler = samplers.CSampler(
            time_step_spec, action_spec, config)

        q_network = tf.make_template(
            'model',
            self._q_network_ctor,
            create_scope_now_=True,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=False,
            config=config)
        self._q_network = q_network()

        self._num_bodies = time_step_spec.observation['position'].shape[0]
        info_spec = {
            'start': tf.TensorSpec(
                [self._num_samples] + list(action_spec['start'].shape),
                tf.float32, 'start'),
            'motion': tf.TensorSpec(
                [self._num_samples] + list(action_spec['motion'].shape),
                tf.float32, 'motion'),
            'state': tf.TensorSpec(
                [self._num_bodies, 2],
                tf.float32, 'state'),
            'pred_state': tf.TensorSpec(
                [self._num_steps, self._num_c_elites, self._num_bodies, 2],
                tf.float32, 'pred_state'),
            'termination': tf.TensorSpec(
                [self._num_steps, self._num_c_elites],
                tf.bool, 'termination'),
            'valid': tf.TensorSpec([self._num_samples], tf.bool, 'valid'),
            'c_id': tf.TensorSpec([self._num_samples], tf.int64, 'c_id'),
        }

        super(SMPCPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            info_spec=info_spec)

    def high_level_plan(self, time_step, seed):
        observation = time_step.observation
        c_samples = self._c_sampler(time_step, self._num_samples, seed)

        outputs = self._q_network.high_level_predict(
            observation, c_samples, use_prune=True)
        rewards_t = outputs['reward']
        decay = tf.cumprod(GAMMA * tf.ones_like(rewards_t), axis=0) / GAMMA
        rewards = tf.reduce_sum(rewards_t * decay, axis=0)
        c_samples = tf.gather(c_samples, outputs['indices'], axis=1)

        # Select final elites.
        _, inds = tf.nn.top_k(rewards, k=self._num_c_elites)
        c_elites = tf.gather(c_samples, inds, axis=1)

        ########
        # Info.
        ########
        state = time_step.observation['position'][..., :2]
        pred_state = outputs['pred_state']['position']
        pred_state = tf.gather(pred_state, inds, axis=1)
        delta_state = pred_state[0] - state
        termination = tf.gather(outputs['termination'], inds, axis=1)
        info = {
            'state': state,
            'pred_state': tf.expand_dims(pred_state, axis=0),
            'termination': tf.expand_dims(termination, axis=0),
        }

        if self._debug:
            top_reward = tf.gather(rewards, inds)
            top_reward_t = tf.gather(rewards_t, inds, axis=1)
            top_reward_t = tf.transpose(top_reward_t, [1, 0])

            top_termination_t = tf.gather(outputs['termination'], inds, axis=1)
            top_termination_t = tf.transpose(top_termination_t, [1, 0])

            print_op = tf.print(
                '-- Final --', '\n',
                'rewards: ', rewards, '\n',
                'mean_rewards: ', tf.reduce_mean(rewards), '\n',
                'top_inds: ', inds, '\n',
                'top_reward: ', top_reward, '\n',
                'top_reward_t: ', top_reward_t, '\n',
                'top_termination_t: ', top_termination_t, '\n',
                'top_delta_state: ', delta_state, '\n',
            )
            with tf.control_dependencies([print_op]):
                c_elites = tf.identity(c_elites)

        return c_elites, info

    def low_level_plan(self, time_step, c_elites, seed):
        observation = time_step.observation
        z_samples = self._z_sampler(time_step, self._num_samples, seed)

        num_tiles = int(self._num_samples / self._num_c_elites)
        c_samples = tf.tile(c_elites, [1, num_tiles, 1])
        c_ids = tf.tile(tf.range(self._num_c_elites, dtype=tf.int64),
                        [num_tiles])

        outputs = self._q_network.low_level_predict(
            observation, z_samples, c_samples, use_prune=True)
        actions = outputs['action']
        rewards_t = outputs['reward']
        decay = tf.cumprod(GAMMA * tf.ones_like(rewards_t), axis=0) / GAMMA
        rewards = tf.reduce_sum(rewards_t * decay, axis=0)

        valid = outputs['valid']
        # c_ids = tf.gather(c_ids, outputs['indices'])

        ########
        # Info.
        ########
        info = {
            'start': actions['start'],
            'motion': actions['motion'],
            'valid': tf.expand_dims(valid, axis=0),
            'c_id': tf.expand_dims(c_ids, axis=0),
        }

        _, inds = tf.nn.top_k(rewards, k=1)
        actions = nest.map_structure(
            lambda x: tf.gather(x, inds, axis=1), actions)

        if self._debug:
            top_reward = tf.gather(rewards, inds)
            top_reward_t = tf.gather(rewards_t, inds, axis=1)
            top_reward_t = tf.transpose(top_reward_t, [1, 0])

            print_op = tf.print(
                '-- Final --', '\n',
                'top_reward: ', top_reward, '\n',
                'top_reward_t: ', top_reward_t, '\n',
                'valid: ', valid, '\n',
            )
            with tf.control_dependencies([print_op]):
                actions = nest.map_structure(lambda x: tf.identity(x), actions)

        return actions, info

    def _variables(self):
        return self._q_network.variables

    def _action(self, time_step, policy_state, seed):
        c_elites, high_info = self.high_level_plan(time_step, seed)
        actions, low_info = self.low_level_plan(time_step, c_elites, seed)
        action = nest.map_structure(lambda x: x[0], actions)
        info = {
            'state': high_info['state'],
            'pred_state': high_info['pred_state'],
            'termination': high_info['termination'],
            'start': low_info['start'],
            'motion': low_info['motion'],
            'valid': low_info['valid'],
            'c_id': low_info['c_id'],
        }
        return policy_step.PolicyStep(action, policy_state, info)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')
