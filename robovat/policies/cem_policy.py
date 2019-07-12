"""Policy with Cross-Entropy Method ."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import policy_step
from tf_agents.policies import tf_policy

nest = tf.contrib.framework.nest
tfd = tfp.distributions


class CemPolicy(tf_policy.Base):
    """Policy with Cross-Entropy Method ."""

    def __init__(self,
                 time_step_spec=None,
                 action_spec=None,
                 critic_network=None,
                 encoding_network=None,
                 initial_sampler=None,
                 num_samples=64,
                 num_elites=6,
                 num_iterations=3,
                 config=None):
        """Builds a Q-Policy given a critic_network.

        Args:
            time_step_spec: A `TimeStep` spec of the expected time_steps.
            action_spec: A nest of BoundedTensorSpec representing the actions.
            critic_network: An instance of a `tf_agents.network.Network`,
                callable via `network(observation, step_type) -> (output,
                final_state)`.

        Raises:
            NotImplementedError: If `action_spec` contains more than one
                `BoundedTensorSpec`.
        """
        flat_action_spec = nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise NotImplementedError(
                    'action_spec can only contain a single BoundedTensorSpec.')
        # We need to extract the dtype shape and shape from the spec while
        # maintaining the nest structure of the spec.
        self._action_dtype = flat_action_spec[0].dtype
        self._action_shape = flat_action_spec[0].shape

        self._critic_network = critic_network
        self._encoding_network = encoding_network
        self._initial_sampler = initial_sampler

        self._num_samples = num_samples
        self._num_elites = num_elites
        self._num_iterations = num_iterations

        self.config = config

        super(CemPolicy, self).__init__(
            time_step_spec,
            action_spec,
            policy_state_spec=critic_network.state_spec)

    def _variables(self):
        if self._encoding_network:
            return (self._critic_network.variables +
                    self._encoding_network.variables)
        else:
            return self._critic_network.variables

    def _action(self, time_step, policy_state, seed):
        if self._initial_sampler:
            initial_samples = self._initial_sampler(
                time_step, policy_state, seed)
        else:
            batch_size = None  # TODO
            initial_samples = tf.random_uniform(
                [batch_size, self._num_samples] + self._action_shape,
                dtype=self._action_dtype)

        if self._encoding_network:
            encoded_state = self._encoding_network(
                time_step.observation, time_step.step_type, policy_state)
        else:
            encoded_state = None

        action = self.cem_loop(
            time_step, policy_state, initial_samples, encoded_state)

        def dist_fn(action):
            return tfp.distributions.Deterministic(loc=action)
        return policy_step.PolicyStep(nest.map_structure(dist_fn, action),
                                      policy_state)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')

    def cem_loop(self,
                 time_step,
                 policy_state,
                 initial_samples,
                 encoded_state=None):
        """Returns an action using CEM."""

        def evaluate(samples):
            if encoded_state:
                return self._critic_network(
                    time_step.observation, time_step.step_type, policy_state,
                    encoded_state, samples)
            else:
                return self._critic_network(
                    time_step.observation, time_step.step_type, policy_state,
                    samples)

        def body(samples, i, num_iterations):
            """Defines the body of the while loop.
            
            Args:
                samples: A tensor of shape [num_samples, action_size].
                i: The iteration index as a tensor.
                num_iterations: Number of iteration.

            Returns: 
                Tuple of new_samples, i + 1, num_iterations.
            """
            scores = evaluate(samples)
            _, ind = tf.nn.top_k(scores, self._num_elites)
            vals = tf.batch_gather(samples, ind)
            mean, var = tf.nn.moments(vals, axes=1)

            new_dist = tf.distributions.Normal(loc=mean, scale=tf.sqrt(var))
            new_samples = new_dist.sample([self._num_samples])
            new_samples = tf.transpose(new_samples, [1, 0, 2])
            return new_samples, tf.add(i, 1), num_iterations

        def cond(samples, i, num_iterations):
            return tf.less(i, num_iterations)

        num_iterations = tf.constant(self._num_iterations)
        samples, _, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[initial_samples, 0, num_iterations])
        final_scores = evaluate(samples)
        _, ind = tf.nn.top_k(final_scores, k=1)
        action = tf.batch_gather(samples, ind)
        action = tf.squeeze(action)

        return action
