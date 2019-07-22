"""Policy with Cross-Entropy Method ."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.policies import policy_step
from tf_agents.policies import tf_policy

from robovat.policies import cem

nest = tf.contrib.framework.nest
tfd = tfp.distributions

class PointCloudPolicy(tf_policy.Base):

    def __init__(self, 
                 time_step_spec=None,
                 action_spec=None,
                 config=None):

        self.config = config
        super(PointCloudPolicy, self).__init__(
                time_step_spec,
                action_spec)

    def evaluate(self, time_step, samples):
        return 

    def _variables(self):
        return 

    def _action(self, time_step, policy_state, seed):
        return

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError
 

class CemPolicy(tf_policy.Base):
    """Policy with cross-entropy method."""

    def __init__(self,
                 time_step_spec=None,
                 action_spec=None,
                 q_network=None,
                 initial_sampler=None,
                 config=None):
        """Builds a Q-Policy given a q_network.

        Args:
            time_step_spec: A `TimeStep` spec of the expected time_steps.
            action_spec: A nest of BoundedTensorSpec representing the actions.
            q_network: An instance of a `tf_agents.network.Network`, callable
                via `network(observation, step_type) -> (output, final_state)`.

        Raises:
            NotImplementedError: If `action_spec` contains more than one
                `BoundedTensorSpec`.
        """
        self._q_network = q_network

        self._cem = cem.CEM(
            self.evaluate,
            initial_sampler=initial_sampler,
            num_samples=config.NUM_SAMPLES,
            num_elites=config.NUM_ELITES,
            num_iterations=config.NUM_ITERATIONS)

        super(CemPolicy, self).__init__(
            time_step_spec,
            action_spec)

    def evaluate(self, time_step, samples):
        return self._q_network(
            time_step.observation,
            samples)

    def _variables(self):
        return self._q_network.variables

    def _action(self, time_step, policy_state, seed):
        action = self._cem(time_step, seed)
        return policy_step.PolicyStep(action, policy_state)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')
