"""Run and generate episodes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket
import time
import traceback

from robovat.utils import time_utils
from robovat.utils.logging import logger


def generate_episode(env, policy, num_steps=None, debug=False):
    """Run and generate an episode.

    Args:
        env: The environment.
        policy: The policy.
        num_steps: Maximum number of steps in each episode. None for infinite.
        debug: True for visualize the policy for debugging, False otherwise.

    Returns:
        episode: The episode data as a dictionary of
            'hostname': The hostname of the computer.
            'timestamp': The system timestamp of the beginning of the episode.
            'policy_info': The policy information.
            'transitions': A list of transitions. Each transition is a
                dictionary of state, action, reward, and info.
    """
    t = 0
    transitions = []

    observation = env.reset()

    while(1):
        action = policy(observation)

        new_observation, reward, done, info = env.step(action)
        transition = {
                'state': observation,
                'action': action,
                'reward': reward,
                'info': info,
                }
        transitions.append(transition)
        observation = new_observation

        if done:
            break

        if (num_steps is not None) and (t >= num_steps):
            break

    episode = {
            'hostname': socket.gethostname(),
            'timestamp': time_utils.get_timestamp_as_string(),
            'env_info': env.info,
            'policy_info': policy.info,
            'transitions': transitions,
            }

    return episode


def generate_episodes(env, policy, num_steps=None, num_episodes=None,
                      timeout=30, debug=False):
    """Run and generate multiple episodes.

    Args:
        env: The environment.
        policy: The policy.
        num_steps: Maximum number of steps in each episode. None for infinite.
        num_episodes: Maximum number of episodes. None for infinite.
        timeout: Seconds to timeout.
        debug: True for visualize the policy for debugging, False otherwise.

    Yields:
        episode_index: The index of the episode.
        episode: The episode data.
    """
    episode_index = 0
    total_time = 0.0

    while(1):
        try:
            tic = time.time()
            if debug:
                episode = generate_episode(env, policy, num_steps, debug)
            else:
                with time_utils.Timeout(timeout):
                    episode = generate_episode(env, policy, num_steps, debug)
            toc = time.time()
            total_time += (toc - tic)

            logger.info(
                    'Episode %d finished in %.2f sec. '
                    'In average each episode takes %.2f sec',
                    episode_index, toc - tic, total_time / (episode_index + 1))

            yield episode_index, episode

            episode_index += 1

            if num_episodes is not None:
                if episode_index >= num_episodes:
                    break

        except Exception as e:
            traceback.print_exc()

            # if env.debug:
            if False:
                exit()
            else:
                logger.error('The episode is discarded due to: %s', type(e))
                env.handle_exception(e)
