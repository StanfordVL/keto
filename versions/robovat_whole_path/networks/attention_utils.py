"""Attention utils.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
        x: a Tensor with shape [..., m]
        n: an integer.
    Returns:
        a Tensor with shape [..., n, m/n]
    """
    x_shape = x.get_shape().as_list()
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


def split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads (becomes dimension 1).
    Args:
        x: a Tensor with shape [batch, length, channels]
        num_heads: an integer
    Returns:
        a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def combine_heads(x):
    """Inverse of split_heads.
    Args:
        x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
    Returns:
        a Tensor with shape [batch, length, channels]
    """
    x = tf.transpose(x, [0, 2, 1, 3])
    x_shape = x.get_shape().as_list()
    a, b = x_shape[-2:]
    return tf.reshape(x, x_shape[:-2] + [a * b])


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          name=None):
    """Dot-product attention.

    Args:
        q: Tensor with shape [..., length_q, depth_k].
        k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
            match with q.
        v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
            match with q.
        bias: bias Tensor (see attention_bias())
        dropout_rate: a float.
        name: an optional string

    Returns:
        Tensor with shape [..., length_q, depth_v].
    """
    with tf.variable_scope(name,
                           default_name='dot_product_attention',
                           values=[q, k, v]):
        # [..., length_q, length_kv]
        logits = tf.matmul(q, k, transpose_b=True)

        if bias is not None:
            bias = tf.cast(bias, logits.dtype)
            logits += bias

        weights = tf.nn.softmax(logits, name='attention_weights')
        weights = tf.cast(weights, q.dtype)

        # Drop out attention links for each head.
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)

        return tf.matmul(weights, v)


def compute_attention_component(antecedent,
                                dim_output,
                                filter_width=1,
                                padding="VALID",
                                name="c"):
    """Computes attention compoenent (query, key or value).

    Args:
        antecedent: a Tensor with shape [batch, length, channels]
        dim_output: an integer
        filter_width: An integer specifying how wide you want the attention
            component to be.
        padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
        name: a string specifying scope name.

    Returns:
        c : [batch, length, depth] tensor
    """
    if filter_width == 1:
        return tf.layers.dense(
                antecedent, dim_output, use_bias=False, name=name)
    else:
        return tf.layers.conv1d(
            antecedent, dim_output, filter_width, padding=padding, name=name)


def compute_qkv(query_antecedent,
                memory_antecedent,
                dim_key,
                dim_value,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding='VALID',
                kv_padding='VALID'):
    """Computes query, key and value.

      Args:
          query_antecedent: a Tensor with shape [batch, length_q, channels]
          memory_antecedent: a Tensor with shape [batch, length_m, channels]
          dim_key: an integer
          dim_value: an integer
          q_filter_width: An integer specifying how wide you want the query to
              be.
          kv_filter_width: An integer specifying how wide you want the keys and
              values to be.
          q_padding: One of 'VALID', 'SAME' or 'LEFT'.
          kv_padding: One of 'VALID', 'SAME' or 'LEFT'.

      Returns:
          q, k, v : [batch, length, depth] tensors
    """
    if memory_antecedent is None:
        memory_antecedent = query_antecedent

    q = compute_attention_component(
            query_antecedent,
            dim_key,
            q_filter_width,
            q_padding,
            'q')
    k = compute_attention_component(
            memory_antecedent,
            dim_key,
            kv_filter_width,
            kv_padding,
            'k')
    v = compute_attention_component(
            memory_antecedent,
            dim_value,
            kv_filter_width,
            kv_padding,
            'v')

    return q, k, v


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        dim_key,
                        dim_value,
                        dim_output,
                        num_heads,
                        dropout_rate,
                        name=None):
    with tf.variable_scope(name, default_name='multihead_attention'):
        # Project inputs to queries and key-value pairs.
        q, k, v = compute_qkv(query_antecedent,
                              memory_antecedent,
                              dim_key,
                              dim_value)
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        # Normalize the queries.
        dim_key_per_head = dim_key // num_heads
        q *= dim_key_per_head**-0.5

        # Dot product attention.
        x = dot_product_attention(q, k, v, bias, dropout_rate)
        x = combine_heads(x)
        x.set_shape(x.shape.as_list()[:-1] + [dim_value])

        # Output layer.
        x = tf.layers.dense(x, dim_output, use_bias=False, name='output')

    return x


def two_layer_feed_forward(inputs,
                           dim_output,
                           activation_fn,
                           normalizer_fn,
                           name=None):
    """The two-layer feed forward network.

    Args:
        inputs: The input tensor.
        dim_output: Dimension of the output.
        activation_fn: The activation function of the output layer.
        normalizer_fn: The normalization function of the output layer.
        name: Name of the layer.

    Returns:
        A float32 tensor of dim_output dimensions.
    """
    with tf.variable_scope(name, default_name='feed_forward'):
        with slim.arg_scope(
            [slim.fully_connected],
            weights_initializer=slim.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False)):
            net = slim.fully_connected(inputs,
                                       dim_output,
                                       activation_fn=tf.nn.relu,
                                       normalizer_fn=slim.layer_norm,
                                       scope='fc1')
            net = slim.fully_connected(net,
                                       dim_output,
                                       activation_fn=activation_fn,
                                       normalizer_fn=normalizer_fn,
                                       scope='fc2')
            return net


def add_and_norm(inputs,
                 outputs,
                 use_relu=True,
                 name=None):
    """Add inputs to outputs and applies layer normalization.

    Args:
        inputs: The input tensor.
        outputs: The comptued output tensor.
        use_relu: Use relu activation if it is True.
        name: Name of the layer.
    """
    with tf.variable_scope(name, default_name='add_and_norm'):
        net = inputs + outputs
        if use_relu:
            net = tf.nn.relu(net)
        net = slim.layer_norm(net)
        return net 


def two_layer_residual_block(inputs, dim_output, scope):
    with tf.variable_scope(scope, default_name='attention_block'):
        short_cut = inputs
        net = two_layer_feed_forward(
            inputs,
            dim_output=dim_output,
            activation_fn=None,
            normalizer_fn=None)
        net = add_and_norm(short_cut, net, name='add_and_norm')
        return net


def attention_block(query_antecedent,
                    memory_antecedent,
                    bias,
                    dim_key,
                    dim_value,
                    num_heads,
                    dropout_rate,
                    name=None):
    """Attention block.

    Args: 
        query_antecedent: An array to be projected as the queries.
        memory_antecedent: An array to be projected as the key-value pairs.
        bias: Bias tensor.
        dim_key: Total dimension of the keys.
        dim_value: Total dimension of the values.
        num_heads: Number of heads of the keys and values.
        dropout_rate: A floating point number.
        name: Name of the layer.
    """
    with tf.variable_scope(name, default_name='attention_block'):
        dim_output = query_antecedent.get_shape()[-1].value

        if bias is not None:
            bias = tf.expand_dims(bias, 1)

        short_cut = query_antecedent
        net = multihead_attention(
            query_antecedent,
            memory_antecedent,
            bias=bias,
            dim_key=dim_key,
            dim_value=dim_value,
            dim_output=dim_output,
            num_heads=num_heads,
            dropout_rate=dropout_rate)
        net = add_and_norm(short_cut, net, name='add_and_norm_1')

        short_cut = net
        net = two_layer_feed_forward(
            net,
            dim_output=dim_output,
            activation_fn=None,
            normalizer_fn=None)
        net = add_and_norm(short_cut, net, name='add_and_norm_2')

        return net
