"""Multi-stage policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt  # NOQA

import numpy as np
import tensorflow as tf

from tf_agents.policies import tf_policy
from tf_agents.policies import policy_step
from tf_agents.specs import tensor_spec

from robovat import networks
from robovat.policies import samplers  # NOQA


nest = tf.contrib.framework.nest


BOUNDARY = 1.

COLLISION_DIST = 0.15
REACHABLE_DIST = 0.4

# COLLISION_DIST = 0.3
# REACHABLE_DIST = 0.6


def plot_actions(prev_movable, new_movable, motion):
    prev_movable = np.squeeze(prev_movable, 0)

    plt.figure(figsize=(5, 5))
    plt.xlim([-1., 1.])
    plt.ylim([-1., 1.])

    # Plot movables.
    for i in range(prev_movable.shape[0]):
        if i == 0:
            color = 'r'
        elif i == 1:
            color = 'g'
        elif i == 2:
            color = 'b'
        else:
            color = 'k'

        plt.scatter(prev_movable[i, 0], prev_movable[i, 1],
                    c=color, s=100, marker='+')

        for j in range(new_movable.shape[0]):
            plt.scatter(new_movable[j, i, 0], new_movable[j, i, 1],
                        c=color, s=10)
            plt.plot([prev_movable[i, 0], new_movable[j, i, 0]],
                     [prev_movable[i, 1], new_movable[j, i, 1]],
                     c=color)

    plt.show()

    return motion


def visualize(prev_movable, new_movable, motion):
    print_ops = [
        tf.py_func(plot_actions, [prev_movable, new_movable, motion],
                   tf.float32),
    ]
    print_op = tf.group(print_ops)
    with tf.control_dependencies([print_op]):
        return tf.identity(motion)


class BrownianPolicy(tf_policy.Base):
    """CEM policy for multi-stage tasks."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None):
        self._num_samples = config.NUM_SAMPLES
        self._sampler = samplers.ZSampler(
            time_step_spec, action_spec, config)

        self._q_network_ctor = getattr(networks, 'BrownianNet')
        q_network = tf.make_template(
            'model',
            self._q_network_ctor,
            create_scope_now_=True,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=False,
            config=config)
        self._q_network = q_network()

        super(BrownianPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec)

    def _variables(self):
        return self._q_network.variables

    def _action(self, time_step, policy_state, seed):
        samples = self._sampler(time_step, self._num_samples, seed)
        outputs = self._q_network(time_step.observation, samples)
        action = outputs['action']

        print_op = tf.print(
            'position: ', action['position'], '\n',
        )
        with tf.control_dependencies([print_op]):
            action = nest.map_structure(lambda x: tf.identity(x), action)

        action['position'] = visualize(
            time_step.observation['movable'], action['position'])

        _policy_step = policy_step.PolicyStep(action, policy_state)
        return _policy_step

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')


class MixtureBrownianPolicy(tf_policy.Base):
    """CEM policy for multi-stage tasks."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None):
        self._num_samples = config.NUM_SAMPLES
        self._sampler = samplers.ZSampler(
            time_step_spec, action_spec, config)

        self._q_network_ctor = getattr(networks, 'MixtureBrownianNetV2')
        q_network = tf.make_template(
            'model',
            self._q_network_ctor,
            create_scope_now_=True,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=False,
            config=config)
        self._q_network = q_network()

        super(MixtureBrownianPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec)

    def _variables(self):
        return self._q_network.variables

    def _action(self, time_step, policy_state, seed):
        samples = self._sampler(time_step, self._num_samples, seed)
        outputs = self._q_network(time_step.observation, samples)

        action = tensor_spec.sample_spec_nest(
            self._action_spec, seed=seed, outer_dims=[1])

        print_op = tf.print(
            'mode: ', outputs['mode'], '\n',
            'mode_prior: ', outputs['mode_prior'], '\n',
            'mode_prior_logit: ', outputs['mode_prior_logit'], '\n',
        )
        with tf.control_dependencies([print_op]):
            action = nest.map_structure(lambda x: tf.identity(x), action)

        action['motion'] = visualize(
            time_step.observation['movable'],
            outputs['next_state']['movable'],
            action['motion'])

        _policy_step = policy_step.PolicyStep(action, policy_state)
        return _policy_step

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')

