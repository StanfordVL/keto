"""Layer utilities.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


batch_norm_params = {
    'decay': 0.997,
    'epsilon': 1e-5,
    'variables_collections': {
                    'beta': None,
                    'gamma': None,
                    'moving_mean': ['moving_vars'],
                    'moving_variance': ['moving_vars'],
                    },
    'center': True,
    'scale': True,
}


def two_layer_feed_forward(inputs,
                           dim_output,
                           activation_fn,
                           normalizer_fn,
                           is_training,
                           name=None):
    """the two-layer feed forward network.

    args:
        inputs: the input tensor.
        dim_output: dimension of the output.
        activation_fn: the activation function of the output layer.
        normalizer_fn: the normalization function of the output layer.
        name: name of the layer.

    returns:
        a float32 tensor of dim_output dimensions.
    """
    with tf.variable_scope(name, default_name='feed_forward'):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):
                net = slim.fully_connected(inputs,
                                           dim_output,
                                           activation_fn=tf.nn.relu,
                                           normalizer_fn=slim.batch_norm,
                                           scope='fc1')
                net = slim.fully_connected(net,
                                           dim_output,
                                           activation_fn=activation_fn,
                                           normalizer_fn=normalizer_fn,
                                           scope='fc2')
                return net


def add_and_norm(inputs,
                 outputs,
                 is_training,
                 name=None):
    """add inputs to outputs and applies layer normalization.

    args:
        inputs: the input tensor.
        outputs: the comptued output tensor.
        use_relu: use relu activation if it is True.
        name: name of the layer.
    """
    with tf.variable_scope(name, default_name='add_and_norm'):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):
                net = inputs + outputs
                net = tf.nn.relu(net)
                net = slim.batch_norm(net)
                return net 


def two_layer_residual_block(
        inputs,
        dim_output,
        is_training,
        scope):
    with tf.variable_scope(scope, default_name='attention_block'):
        short_cut = inputs
        net = two_layer_feed_forward(
            inputs,
            dim_output=dim_output,
            activation_fn=None,
            normalizer_fn=None,
            is_training=is_training)
        net = add_and_norm(
            short_cut,
            net,
            is_training=is_training,
            name='add_and_norm')
        return net
