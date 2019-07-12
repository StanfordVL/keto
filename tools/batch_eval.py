#!/usr/bin/env python

"""Evaluate the network using TF-Slim.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import math
import os

import cv2  # NOQA
import tensorflow as tf
from tf_agents.environments import tf_py_environment

import _init_paths  # NOQA
from robovat import networks
from robovat import problems
from robovat.dataset.dataset import provide_batch
from robovat.envs import suite_env
from robovat.simulation.simulator import Simulator
from robovat.utils.logging import logger
from robovat.utils.yaml_config import YamlConfig


slim = tf.contrib.slim
tf.set_random_seed(42)


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
        '--env_config',
        dest='env_config',
        type=str,
        help='The configuration file for the environment.',
        default=None)

    parser.add_argument(
        '--problem',
        dest='problem',
        type=str,
        help='Name of the problem.',
        required=True)

    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        type=str,
        help='Path to the training set.',
        required=True)

    parser.add_argument(
        '--network',
        dest='network',
        type=str,
        help='Name of the network.',
        required=True)

    parser.add_argument(
        '--working_dir',
        dest='working_dir',
        type=str,
        help='The working directory.',
        required=True)

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        type=int,
        help='The batch size.',
        default=128)

    parser.add_argument(
        '--num_threads',
        dest='num_threads',
        type=int,
        help='Number of threads for reading data.',
        default=4)

    parser.add_argument(
        '--num_examples',
        dest='num_examples',
        type=int,
        help='Number of examples to evaluate.',
        default=10000)

    parser.add_argument(
        '--eval_interval_secs',
        dest='eval_interval_secs',
        type=int,
        help='The frequency to evaluate model, in seconds.',
        default=60)

    parser.add_argument(
        '--use_cpu',
        dest='use_cpu',
        type=int,
        help='If True, run evaluation on CPU.',
        default=1)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Process the directories.
    if args.working_dir is not None:
        if not os.path.exists(args.working_dir):
            os.makedirs(args.working_dir)

    g = tf.Graph()
    with g.as_default():
        # Environment.
        if args.env is None:
            time_step_spec = ()
            action_spec = ()
        else:
            simulator = Simulator()

            if args.env_config is None:
                env_config = None
            else:
                env_config = YamlConfig(args.env_config).as_easydict()

            py_env = suite_env.load(args.env,
                                    simulator=simulator,
                                    config=env_config,
                                    debug=False,
                                    max_episode_steps=None)
            tf_env = tf_py_environment.TFPyEnvironment(py_env)
            time_step_spec = tf_env.time_step_spec()
            action_spec = tf_env.action_spec()
            del tf_env
            del simulator

        # Problem.
        problem_cls = getattr(problems, args.problem)
        problem = problem_cls(time_step_spec=time_step_spec,
                              action_spec=action_spec,
                              is_training=False)

        # Dataset.
        data_path = os.path.join(args.data_dir, '*.tfrecords')
        num_data_files = len(glob.glob(data_path))
        logger.info(
            'Loading training set from %s (%d files) with batch size of %d...'
            % (data_path, num_data_files, args.batch_size))
        batch = provide_batch(filename=data_path,
                              problem=problem,
                              batch_size=args.batch_size,
                              num_threads=args.num_threads,
                              shuffle=False)
        batch = problem.preprocess(batch)

        # Network.
        logger.info('Building the neural network...')
        if args.policy_config is None:
            policy_config = None
        else:
            policy_config = YamlConfig(args.policy_config).as_easydict()
        network_cls = getattr(networks, args.network)
        network = network_cls(time_step_spec=time_step_spec,
                              action_spec=action_spec,
                              config=policy_config,
                              is_training=False)
        outputs = network.forward(batch)

        loss = problem.loss(batch, outputs)

        # Evaluation.
        slim.get_or_create_global_step()
        num_batches = math.ceil(args.num_examples / float(args.batch_size))

        logger.info('Adding summaries...')
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tf.summary.scalar('loss', loss)
        with tf.name_scope('variables'):
            for var in variables:
                tf.summary.histogram(var.name, var)
        eval_ops = problem.add_summaries(batch, outputs)

        if args.use_cpu:
            session_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            session_config = tf.ConfigProto()

        # Run evaluation.
        if os.path.isdir(args.working_dir):
            logger.info('Run evaluation loop...')
            slim.evaluation.evaluation_loop(
                master='',
                checkpoint_dir=os.path.join(args.working_dir, 'train'),
                logdir=os.path.join(args.working_dir, 'eval'),
                num_evals=num_batches,
                eval_op=eval_ops,
                eval_interval_secs=args.eval_interval_secs,
                session_config=session_config)
        else:
            logger.info('Run evaluation for checkpoint %s...'
                        % (args.working_dir))
            slim.evaluation.evaluate_once(
                master='',
                checkpoint_path=os.path.join(args.working_dir, 'train'),
                logdir=os.path.join(args.working_dir, 'eval'),
                num_evals=num_batches,
                eval_op=eval_ops)


if __name__ == '__main__':
    main()
