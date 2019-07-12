#!/usr/bin/env python

"""Train the network using TF-Slim.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os

import cv2  # NOQA
import tensorflow as tf
from tensorflow.contrib.gan.python import namedtuples as gan_namedtuples
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
tfgan = tf.contrib.gan
tf.set_random_seed(42)


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num_modes',
        dest='num_modes',
        type=int,
        help='TODO',
        default=None)

    parser.add_argument(
        '--env',
        dest='env',
        type=str,
        help='The environment name.',
        default=None)

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
        '--policy_config',
        dest='policy_config',
        type=str,
        help='The configuration file for the policy.',
        default=None)

    parser.add_argument(
        '--working_dir',
        dest='working_dir',
        type=str,
        help='The working directory.',
        required=True)

    parser.add_argument(
        '--pretrained',
        dest='pretrained_dir',
        type=str,
        help='Directory to the pretrained model.',
        default=None)

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
        '--lr',
        dest='learning_rate',
        type=float,
        help='The learning rate.',
        default=1e-3)

    parser.add_argument(
        '--clip_gradient_norm',
        dest='clip_gradient_norm',
        type=float,
        help='If greater than 0 then the gradients would be clipped by it.',
        default=-1)

    parser.add_argument(
        '--save_summaries_secs',
        dest='save_summaries_secs',
        type=int,
        help='The frequency to save summaries, in seconds.',
        default=60)

    parser.add_argument(
        '--save_interval_secs',
        dest='save_interval_secs',
        type=int,
        help='The frequency to save model, in seconds.',
        default=300)

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
                              is_training=True)

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
                              shuffle=True)
        batch = tf.train.shuffle_batch(
                batch,
                batch_size=args.batch_size,
                capacity=50000,
                min_after_dequeue=5000,
                num_threads=args.num_threads,
                enqueue_many=True)
        batch = problem.preprocess(batch)

        # Network.
        logger.info('Building the neural network...')
        if args.policy_config is None:
            policy_config = None
        else:
            policy_config = YamlConfig(args.policy_config).as_easydict()

        # TODO
        if args.num_modes is not None:
            policy_config['ACTION']['NUM_MODES'] = args.num_modes

        network_cls = getattr(networks, args.network)
        network = network_cls(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            config=policy_config,
            is_training=True)
        train_op = problem.get_train_op(
            network,
            batch,
            learning_rate=args.learning_rate,
            clip_gradient_norm=args.clip_gradient_norm)

        # Load the pretrained model.
        if args.pretrained_dir is None:
            init_fn = None
        else:
            train_dir = os.path.join(args.pretrained_dir, 'train')
            pretrained_path = tf.train.latest_checkpoint(train_dir)

            if pretrained_path is None:
                raise ValueError('Could not find any checkpoint file in %s.'
                                 % (train_dir))
            else:
                logger.info('Loading the checkpoint %s...', pretrained_path)

            def init_fn(sess):
                restorer = tf.train.Saver(name='restorer')
                restorer.restore(sess, pretrained_path)

        # Run training.
        if isinstance(train_op, gan_namedtuples.GANTrainOps):
            logger.info('Run GAN training...')
            tfgan.gan_train(
                train_op,
                os.path.join(args.working_dir, 'train'),
                save_summaries_steps=100,
                save_checkpoint_secs=args.save_interval_secs)
        else:
            logger.info('Run training...')
            slim.learning.train(
                train_op,
                os.path.join(args.working_dir, 'train'),
                init_fn=init_fn,
                save_summaries_secs=args.save_summaries_secs,
                save_interval_secs=args.save_interval_secs)


if __name__ == '__main__':
    main()
