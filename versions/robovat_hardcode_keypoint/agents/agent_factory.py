"""Factory for building the agent.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from robovat.utils.string_utils import snakecase_to_camelcase


def get_agent_class(agent_name):
    """Returns an agent class.

    Args:
        agent_name: The name of the agent.

    Returns:
        agent_class: A class of the agent.
    """
    from robovat import agents
    agent_name = '%s_agent' % (agent_name)
    agent_class_name = snakecase_to_camelcase(agent_name)
    agent_class = getattr(agents, agent_class_name)
    return agent_class

