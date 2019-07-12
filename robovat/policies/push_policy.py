"""Multi-stage policies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_step

from robovat.policies import cem_policy  # NOQA
from robovat.policies import samplers  # NOQA

nest = tf.contrib.framework.nest


class HeuristicsPushPolicy(random_tf_policy.RandomTFPolicy):
    """Random policy using modes"""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None):
        self._debug = config.DEBUG
        self._sampler = samplers.HeuristicSampler(
             time_step_spec=time_step_spec,
             action_spec=action_spec,
             config=config)

        super(HeuristicsPushPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec)

    def _action(self, time_step, policy_state, seed):
        action = self._sampler(time_step, 1, seed)

        # Debug
        # if self._debug:
        #     print_op = tf.print(
        #         'start: ', action['start'], '\n',
        #         'motion: ', action['motion'], '\n',
        #     )
        #     with tf.control_dependencies([print_op]):
        #         action = nest.map_structure(lambda x: tf.identity(x), action)

        return policy_step.PolicyStep(action, policy_state)
