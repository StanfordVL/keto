"""Factory for building the environment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from robovat.utils.string_utils import snakecase_to_camelcase


def get_env_class(env_name):
    """Returns a policy class.

    Args:
        env_name: The name of the environment.

    Returns:
        env_class: A class of the environment.
    """
    from robovat import envs
    env_name = '%s_env' % (env_name)
    env_class_name = snakecase_to_camelcase(env_name)
    env_package = getattr(envs, env_name)
    env_class = getattr(env_package, env_class_name)
    return env_class
