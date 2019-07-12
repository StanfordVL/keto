"""A Cross-Entropy Method (CEM) Agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.dqn.dqn_agent import DqnLossInfo
from tf_agents.agents.dqn.dqn_agent import compute_td_targets
from tf_agents.agents.dqn.dqn_agent import element_wise_huber_loss
from tf_agents.environments import trajectory
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import q_policy
from tf_agents.utils import common as common_utils
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils

import gin.tf

nest = tf.contrib.framework.nest


@gin.configurable
class CemAgent(tf_agent.TFAgent):
    """A Cross-Entropy Method (CEM) Agent."""

    def __init__(
            self,
            time_step_spec,
            action_spec,
            q_network,
            optimizer,
            epsilon_greedy=0.1,
            # Params for target network updates
            target_update_tau=1.0,
            target_update_period=1,
            # Params for training.
            td_errors_loss_fn=None,
            gamma=1.0,
            reward_scale_factor=1.0,
            gradient_clipping=None,
            # Params for debugging
            debug_summaries=False,
            summarize_grads_and_vars=False):
        """Creates a DQN Agent.

        Args:
            time_step_spec: A `TimeStep` spec of the expected time_steps.
            action_spec: A nest of BoundedTensorSpec representing the actions.
            q_network: A tf_agents.network.Network to be used by the agent. The
                network will be called with call(observation, step_type).
            optimizer: The optimizer to use for training.
            epsilon_greedy: probability of choosing a random action in the
                default epsilon-greedy collect policy (used only if a wrapper is
                not provided to the collect_policy method).
            target_update_tau: Factor for soft update of the target networks.
            target_update_period: Period for soft update of the target networks.
            td_errors_loss_fn: A function for computing the TD errors loss. If
                None, a default value of element_wise_huber_loss is used. This
                function takes as input the target and the estimated Q values
                and returns the loss for each element of the batch.
            gamma: A discount factor for future rewards.
            reward_scale_factor: Multiplicative scale for the reward.
            gradient_clipping: Norm length to clip gradients.
            debug_summaries: A bool to gather debug summaries.
            summarize_grads_and_vars: If True, gradient and network variable
                summaries will be written during training.

        Raises:
            ValueError: If the action spec contains more than one action or
                action spec minimum is not equal to 0.
        """
        flat_action_spec = nest.flatten(action_spec)
        self._num_actions = [
                spec.maximum - spec.minimum + 1 for spec in flat_action_spec
        ]

        # TODO(oars): Get DQN working with more than one dim in the actions.
        if len(flat_action_spec) > 1 or flat_action_spec[0].shape.ndims > 1:
            raise ValueError('Only one dimensional actions are supported now.')

        if not all(spec.minimum == 0 for spec in flat_action_spec):
            raise ValueError(
                'Action specs should have minimum of 0, but saw: {0}'.format(
                    [spec.minimum for spec in flat_action_spec]))

        self._q_network = q_network
        self._target_q_network = self._q_network.copy(name='TargetQNetwork')
        self._epsilon_greedy = epsilon_greedy
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period
        self._optimizer = optimizer
        self._td_errors_loss_fn = td_errors_loss_fn or element_wise_huber_loss
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._gradient_clipping = gradient_clipping

        self._target_update_train_op = None

        policy = q_policy.QPolicy(
                time_step_spec, action_spec, q_network=self._q_network)

        collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
                policy, epsilon=self._epsilon_greedy)
        policy = greedy_policy.GreedyPolicy(policy)

        super(CemAgent, self).__init__(
                time_step_spec,
                action_spec,
                policy,
                collect_policy,
                train_sequence_length=2 if not q_network.state_spec else None,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars)

    def _initialize(self):
        return self._update_targets(1.0, 1)

    def _update_targets(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.
        For each weight w_s in the q network, and its corresponding
        weight w_t in the target_q_network, a soft update is:
        w_t = (1 - tau) * w_t + tau * w_s

        Args:
            tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
            period: Step interval at which the target network is updated.

        Returns:
            An operation that performs a soft update of the target network
                parameters.
        """
        with tf.name_scope('update_targets'):

            def update():
                return common_utils.soft_variables_update(
                    self._q_network.variables,
                    self._target_q_network.variables,
                    tau)

            return common_utils.periodically(
                update, period, 'periodic_update_targets')

    def _experience_to_transitions(self, experience):
        transitions = trajectory.to_transition(experience)

        # Remove time dim if we are not using a recurrent network.
        if not self._q_network.state_spec:
            transitions = nest.map_structure(
                lambda x: tf.squeeze(x, [1]), transitions)

        time_steps, policy_steps, next_time_steps = transitions
        actions = policy_steps.action
        return time_steps, actions, next_time_steps

    def _train(self, experience, train_step_counter=None, weights=None):
        time_steps, actions, next_time_steps = self._experience_to_transitions(
                experience)

        loss_info = self._loss(
                time_steps,
                actions,
                next_time_steps,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights)

        transform_grads_fn = None
        if self._gradient_clipping is not None:
            transform_grads_fn = tf.contrib.training.clip_gradient_norms_fn(
                    self._gradient_clipping)

        loss_info = eager_utils.create_train_step(
                loss_info,
                self._optimizer,
                total_loss_fn=lambda loss_info: loss_info.loss,
                global_step=train_step_counter,
                transform_grads_fn=transform_grads_fn,
                summarize_gradients=self._summarize_grads_and_vars,
                variables_to_train=lambda: self._q_network.trainable_weights,
        )

        if isinstance(loss_info, eager_utils.Future):
            loss_info = loss_info()

        # Make sure the update_targets periodically object is only created once.
        if self._target_update_train_op is None:
            with tf.control_dependencies([loss_info.loss]):
                self._target_update_train_op = self._update_targets(
                        self._target_update_tau, self._target_update_period)

        with tf.control_dependencies([self._target_update_train_op]):
            loss_info = nest.map_structure(
                    lambda t: tf.identity(t, name='loss_info'), loss_info)

        return loss_info

    @eager_utils.future_in_eager_mode
    # TODO(b/79688437): Figure out how to enable defun for Eager mode.
    # @tfe.defun
    def _loss(self,
              time_steps,
              actions,
              next_time_steps,
              td_errors_loss_fn=element_wise_huber_loss,
              gamma=1.0,
              reward_scale_factor=1.0,
              weights=None):
        """Computes loss for DQN training.

        Args:
            time_steps: A batch of timesteps.
            actions: A batch of actions.
            next_time_steps: A batch of next timesteps.
            td_errors_loss_fn: A function(td_targets, predictions) to compute
            the element wise loss.
            gamma: Discount for future rewards.
            reward_scale_factor: Multiplicative factor to scale rewards.
            weights: Optional scalar or elementwise (per-batch-entry) importance
                weights. The output td_loss will be scaled by these weights, and
                the final scalar loss is the mean of these values.

        Returns:
            loss: An instance of `DqnLossInfo`.

        Raises:
            ValueError:
                if the number of actions is greater than 1.
        """
        with tf.name_scope('loss'):
            actions = nest.flatten(actions)[0]
            q_values, _ = self._q_network(time_steps.observation,
                                          time_steps.step_type,
                                          actions)

            # TODO(kuanfang): Compute the next Q values using QT-Opt.
            next_q_values = tf.zeros_like(q_values)
            td_targets = compute_td_targets(
                    next_q_values,
                    rewards=reward_scale_factor * next_time_steps.reward,
                    discounts=gamma * next_time_steps.discount)

            valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
            td_error = valid_mask * (td_targets - q_values)
            td_loss = valid_mask * td_errors_loss_fn(td_targets, q_values)

            if nest_utils.is_batched_nested_tensors(
                    time_steps, self.time_step_spec(), num_outer_dims=2):
                # Do a sum over the time dimension.
                td_loss = tf.reduce_sum(td_loss, axis=1)

            if weights is not None:
                td_loss *= weights

            loss = tf.reduce_mean(td_loss)

            with tf.name_scope('Losses/'):
                tf.contrib.summary.scalar('loss', loss)

            if self._summarize_grads_and_vars:
                with tf.name_scope('Variables/'):
                    for var in self._q_network.trainable_weights:
                        tf.contrib.summary.histogram(
                            var.name.replace(':', '_'), var)

            if self._debug_summaries:
                diff_q_values = q_values - next_q_values
                common_utils.generate_tensor_summaries('td_error', td_error)
                common_utils.generate_tensor_summaries('td_loss', td_loss)
                common_utils.generate_tensor_summaries('q_values', q_values)
                common_utils.generate_tensor_summaries('next_q_values',
                                                       next_q_values)
                common_utils.generate_tensor_summaries('diff_q_values',
                                                       diff_q_values)

            return tf_agent.LossInfo(
                loss, DqnLossInfo(td_loss=td_loss, td_error=td_error))
