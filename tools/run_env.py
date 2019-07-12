#!/usr/bin/env python

"""Run an environment with the chosen policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy  # NOQA

import _init_paths  # NOQA
from robovat import policies
from robovat.envs import suite_env
from robovat.envs import py_driver
from robovat.simulation.simulator import Simulator
from robovat.utils.logging import logger
from robovat.utils.yaml_config import YamlConfig


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env',
        dest='env',
        type=str,
        help='The environment name.',
        required=True)

    parser.add_argument(
        '--use_simulator',
        dest='use_simulator',
        type=bool,
        help='Run experiments in the simulation is it is True.',
        required=False,
        default=True)

    parser.add_argument(
        '--worker_id',
        dest='worker_id',
        type=int,
        help='The worker ID for running multiple simulations in parallel.',
        default=0)

    parser.add_argument(
        '--timeout',
        dest='timeout_steps',
        type=int,
        help='Maximum number of simulation steps.',
        required=False,
        default=None)

    parser.add_argument(
        '--max_steps',
        dest='max_steps',
        type=int,
        help='Maximum number of time steps for each episode.',
        default=None)

    parser.add_argument(
        '--num_episodes',
        dest='num_episodes',
        type=int,
        help='Maximum number of episodes.',
        default=None)

    parser.add_argument(
        '--policy',
        dest='policy',
        type=str,
        help='The policy name.',
        required=True)

    parser.add_argument(
        '--policy_config',
        dest='policy_config',
        type=str,
        help='The configuration file for the policy.',
        required=False,
        default=None)

    parser.add_argument(
        '--debug',
        dest='debug',
        type=int,
        help='Use the debugging mode if it is True.',
        default=0)

    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='None for random; any fixed integers for deterministic.',
        default=None)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Set the random seed.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    # Simulator.
    if args.use_simulator:
        simulator = Simulator(worker_id=args.worker_id,
                              max_steps=args.timeout_steps,
                              use_visualizer=bool(args.debug))
    else:
        simulator = None

    # Environment.
    logger.info('Building the environment %s...', args.env)

    py_env = suite_env.load(args.env,
                            simulator=simulator,
                            config=None,
                            debug=args.debug,
                            max_episode_steps=args.max_steps)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    policy_class = getattr(policies, args.policy)
    policy_config = YamlConfig(args.policy_config).as_easydict()
    tf_policy = policy_class(time_step_spec=tf_env.time_step_spec(),
                             action_spec=tf_env.action_spec(),
                             config=policy_config,
                             debug=args.debug)

    py_policy = py_tf_policy.PyTFPolicy(tf_policy)

    with tf.Session():
        # Generate episodes.
        time_step = py_env.reset()
        policy_state = py_policy.get_initial_state(py_env.batch_size)
        driver = py_driver.PyDriver(
            py_env,
            py_policy,
            observers=[],
            max_episodes=args.num_episodes)
        driver.run(time_step, policy_state)


if __name__ == '__main__':
    main()
