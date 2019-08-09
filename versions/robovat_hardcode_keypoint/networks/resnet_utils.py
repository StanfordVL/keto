"""Resnet model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def instance_norm_relu(inputs):
    """Performs an instance normalization followed by a ReLU.
    """
    net = tf.contrib.layers.instance_norm(
        inputs=inputs, epsilon=1e-5, center=True, scale=True)
    net = tf.nn.relu(net)
    return net


def fixed_padding(inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        kernel_size: The kernel to be used in the conv2d or max_pool2d
            operation. Should be a positive integer.
    Returns:
        A tensor with the same format as the input with the data either intact
            (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])

    return padded_inputs


def conv2d_fixed_padding(inputs, num_outputs, kernel_size, stride,
                         trainable=True, scope=None):
    """Strided 2-D convolution with explicit padding.
    """
    # The padding is consistent and is based only on `kernel_size`, not on the 
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if stride > 1:
        inputs = fixed_padding(inputs, kernel_size)

    return tf.layers.conv2d(
        inputs, num_outputs, kernel_size, strides=stride,
        padding=('SAME' if stride == 1 else 'VALID'),
        activation=None,
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        trainable=trainable,
        name=scope)


def projection_shortcut(inputs, num_outputs, stride, trainable=True,
                        scope=None):
    """Projection shortcut layer.
    """
    return conv2d_fixed_padding(inputs=inputs, num_outputs=num_outputs,
                                kernel_size=1, stride=stride,
                                trainable=trainable, scope=scope)


def resnet_block(inputs, num_outputs, stride, shortcut_fn=None, trainable=True,
                 scope=None):
    """Standard building block for residual networks.

    Args:
        inputs: A tensor of shape [batch, height, width, channels].
        num_outputs: The number of filters for the convolutions.
        shortcut_fn: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
        stride: The block's stride. If greater than 1, this block will
            ultimately downsample the input.
      scope: A string name for the tensor output of the block.

    Returns:
        The output tensor of the block.
    """
    with tf.variable_scope(scope):
        # The projection shortcut should come after the first batch norm and
        # ReLU since it performs a 1x1 convolution.
        if shortcut_fn is not None:
            shortcut = shortcut_fn(inputs, num_outputs, stride,
                                   trainable=trainable, scope='shortcut')
        else:
            shortcut = tf.identity(inputs)

        net = conv2d_fixed_padding(inputs=inputs, num_outputs=num_outputs,
                                   kernel_size=3, stride=stride,
                                   trainable=trainable, scope='conv1')

        net = instance_norm_relu(net)

        net = conv2d_fixed_padding(inputs=net, num_outputs=num_outputs,
                                   kernel_size=3, stride=1,
                                   trainable=trainable, scope='conv2')

        net = instance_norm_relu(net + shortcut)

    return net


def resnet_block_layer(inputs, num_outputs, stride, num_blocks=1,
                       shortcut_fn=None, trainable=True, scope=None):
    """Creates one layer of blocks for the ResNet model.

    Args:
      inputs: A tensor of shape [batch, height, width, channels].
      num_outputs: The number of filters for the first convolution of the layer.
      stride: The stride to use for the first convolution of the layer. If
          greater than 1, this layer will ultimately downsample the input.
      num_blocks: The number of blocks contained in the layer.
      shortcut_fn: The shortcut function.
      scope: A string name for the tensor output of the block layer.

    Returns:
      The output tensor of the block layer.
    """
    if shortcut_fn == 'projection':
        shortcut_fn = projection_shortcut

    with tf.variable_scope(scope):
        # Only the first block uses projection_shortcut and strides.
        net = resnet_block(inputs, num_outputs, stride, shortcut_fn, 
                           trainable=trainable,
                           scope='block1')

        for block_id in range(1, num_blocks):
            net = resnet_block(net, num_outputs, 1, None, 
                               trainable=trainable,
                               scope='block%d' % (block_id + 1))

        return net
