"""Base class of problem.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc

import tensorflow as tf

from robovat.utils.logging import logger


slim = tf.contrib.slim


class Problem(object):
    """Base class of problem."""

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training):
        """Initialize."""
        self.time_step_spec = time_step_spec
        self.action_spec = action_spec
        self.is_training = is_training

    @property
    def spec(self):
        """A nest of `TensorSpec` representing the inputs and the targets."""
        return self._spec

    def convert_trajectory(self, trajectory):
        """Convert trajectory."""
        raise NotImplementedError

    def preprocess(self, batch):
        """Augment batched point cloud.

        Args:
            batch_data: Tensor of shape [batch_size, num_points, channels].

        Returns:
            Tensor of shape [batch_size, num_points, channels].
        """
        return batch

    def loss(self, targets, outputs):
        """Get the loss function."""
        raise NotImplementedError

    def add_summaries(self, targets, outputs):
        """Add summaries."""
        return [tf.no_op()]

    def get_learning_rate(self,
                          learning_rate,
                          decay_steps=400000,
                          decay_rate=0.1):
        return tf.train.exponential_decay(
                  learning_rate,
                  slim.get_or_create_global_step(),
                  decay_steps,
                  decay_rate,
                  staircase=True,
                  name='learning_rate')

    def get_train_op(self,
                     network,
                     batch,
                     learning_rate,
                     clip_gradient_norm=-1):
        logger.info('Forwarding network...')
        outputs = network.forward(batch)

        logger.info('Building the loss function...')
        loss = self.loss(batch, outputs)

        logger.info('Adding summaries...')
        tf.summary.scalar('lr', learning_rate)
        tf.summary.scalar('loss', loss)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.name_scope('variables'):
            for var in variables:
                tf.summary.histogram(var.name, var)
        eval_ops = self.add_summaries(batch, outputs)

        logger.info('Building the train_op...')
        slim.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        learning_rate = self.get_learning_rate(learning_rate)
        train_op = slim.learning.create_train_op(
                loss,
                optimizer,
                clip_gradient_norm=clip_gradient_norm,
                summarize_gradients=True)
        train_op = tf.group(train_op, *eval_ops)

        return train_op
