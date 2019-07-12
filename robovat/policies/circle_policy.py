"""Multi-stage policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt  # NOQA

import numpy as np
import tensorflow as tf

from tf_agents.policies import tf_policy
from tf_agents.policies import policy_step

from robovat import networks
from robovat.policies import samplers  # NOQA


nest = tf.contrib.framework.nest


BOUNDARY = 1.

COLLISION_DIST = 0.15
REACHABLE_DIST = 0.4

# COLLISION_DIST = 0.3
# REACHABLE_DIST = 0.6


def plot_actions(movable, positions):
    movable = np.squeeze(movable, 0)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    plt.xlim([-BOUNDARY, BOUNDARY])
    plt.ylim([-BOUNDARY, BOUNDARY])

    # Plot movables.
    for i in range(movable.shape[0]):
        pos = movable[i]
        circle1 = plt.Circle((pos[0], pos[1]),
                             REACHABLE_DIST,
                             color='g', alpha=.2, fill=True)
        circle2 = plt.Circle((pos[0], pos[1]),
                             COLLISION_DIST,
                             color='r', alpha=.2, fill=True)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        plt.text(pos[0], pos[1], '%d' % i)

    # Plot actions.
    for i in range(positions.shape[0]):
        position = positions[i]
        dists = np.linalg.norm(movable - position, axis=-1)
        is_reachable = dists <= REACHABLE_DIST
        is_reachable = np.any(is_reachable)
        is_collided = dists <= COLLISION_DIST
        is_collided = np.any(is_collided)
        is_action_valid = is_reachable and (not is_collided)

        action_color = 'g' if is_action_valid else 'r'
        plt.scatter(position[0], position[1], c=action_color, marker='+', s=100)

    plt.show()

    return positions


def visualize(movable, position):
    print_ops = [
        tf.py_func(plot_actions, [movable, position], tf.float32),
    ]
    print_op = tf.group(print_ops)
    with tf.control_dependencies([print_op]):
        return tf.identity(position)


class CirclePolicy(tf_policy.Base):
    """CEM policy for multi-stage tasks."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None):
        self._num_samples = config.NUM_SAMPLES
        self._sampler = samplers.ZSampler(
            time_step_spec, action_spec, config)

        self._q_network_ctor = getattr(networks, 'CircleNet')
        q_network = tf.make_template(
            'model',
            self._q_network_ctor,
            create_scope_now_=True,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=False,
            config=config)
        self._q_network = q_network()

        super(CirclePolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec)

    def _variables(self):
        return self._q_network.variables

    def _action(self, time_step, policy_state, seed):
        samples = self._sampler(time_step, self._num_samples, seed)
        outputs = self._q_network(time_step.observation, samples)
        action = outputs['action']

        # action['position'] = visualize(
        #     time_step.observation['movable'], action['position'])

        _policy_step = policy_step.PolicyStep(action, policy_state)
        return _policy_step

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')


class MixtureCirclePolicy(tf_policy.Base):
    """CEM policy for multi-stage tasks."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None):
        self._num_samples = config.NUM_SAMPLES
        self._sampler = samplers.ZSampler(
            time_step_spec, action_spec, config)

        self._q_network_ctor = getattr(networks, 'MixtureCircleNet')
        q_network = tf.make_template(
            'model',
            self._q_network_ctor,
            create_scope_now_=True,
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=False,
            config=config)
        self._q_network = q_network()

        super(MixtureCirclePolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec)

    def _variables(self):
        return self._q_network.variables

    def _action(self, time_step, policy_state, seed):
        samples = self._sampler(time_step, self._num_samples, seed)
        outputs = self._q_network(time_step.observation, samples)
        action = outputs['action']

        print_op = tf.print(
            'mode: ', outputs['mode'], '\n',
            'mode_prior: ', outputs['mode_prior'], '\n',
            'mode_prior_logit: ', outputs['mode_prior_logit'], '\n',
            'position: ', action['position'], '\n',
        )
        with tf.control_dependencies([print_op]):
            action = nest.map_structure(lambda x: tf.identity(x), action)

        # action['position'] = visualize(
        #     time_step.observation['movable'], action['position'])

        return policy_step.PolicyStep(action, policy_state)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')
