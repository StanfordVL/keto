"""Latent Dynamics Model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf
from tf_agents.networks import network

from robovat.utils.logging import logger  # NOQA

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


class ModeClassifier(network.Network):
    """Factorized Latent Dynamics Model."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 use_structured_action=False,
                 name='model'):
        """Initialize."""
        super(FLDM, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training,
            config=config,
            use_structured_action=use_structured_action,
            name=name)

        self.dim_z = config.DIM_Z
        self.num_modes = config.ACTION['NUM_MODES']

    def forward(self, batch):
        """Forward the network in the policy or on the dataset."""
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            outputs = dict()

            state, action, next_state = self.get_inputs_from_batch(batch)
            # state: [batch_size, shape_state]
            # action: [batch_size, shape_action]
            # next_state: [batch_size, shape_state]

            ####
            # Dynamics Network.
            ####
            feature = self.state_encoder(state)
            outputs['feature'] = feature
            # [batch_size, dim_state_feature]

            state_decoded = self.state_decoder(feature)
            outputs['state'] = state_decoded
            # [batch_size, shape_state]

            next_feature = self.next_state_encoder(next_state)
            outputs['next_feature'] = next_feature
            # [batch_size, num_modes, dim_state_feature]

            mode_prob, mode_logit = self.mode_classifier(feature, next_feature)
            outputs['mode_prob'] = mode_prob
            outputs['mode_logit'] = mode_logit
            # [num_samples, num_modes]

            return outputs

    def state_encoder(self, state):
        with tf.variable_scope('state_encoder'):
            return self._state_encoder(state)

    def next_state_encoder(self, state):
        with tf.variable_scope('next_state_encoder'):
            return self._state_encoder(state)

    def mode_classifier(self, feature, next_feature):
        with tf.variable_scope('mode_classifier'):
            return self._mode_classifier(feature, next_feature)

    def _state_encoder(self, state):
        with slim.arg_scope(
                [slim.fully_connected],
                weights_initializer=(
                    tf.contrib.layers.xavier_initializer()),
                weights_regularizer=slim.l2_regularizer(0.0005),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.layer_norm):

            net = state['pose']
            net = slim.fully_connected(net, self.dim_center_fc, scope='fc1')
            net = slim.fully_connected(net, self.dim_center_fc, scope='fc2')
            feature = slim.fully_connected(
                net,
                self.dim_center_fc,
                activation_fn=None,
                normalizer_fn=None,
                scope='feature')

            return feature

    def _mode_classifier(self, feature, next_feature):
        with slim.arg_scope(
                [slim.fully_connected],
                weights_initializer=(
                    tf.truncated_normal_initializer(0.0, 0.01)),
                weights_regularizer=slim.l2_regularizer(0.0005),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.layer_norm):

            delta_feature = tf.identity(
                next_feature - feature, name='delta_feature')

            anchors = tf.get_variable(
                'anchors',
                [self.num_modes, self.dim_state_fc],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(0.0, 0.01))
            anchors = tf.tile(
                tf.expand_dims(anchors, 0),
                [int(delta_feature.get_shape()[0]), 1, 1])

            attn_output = attention_utils.attention_block(
                anchors,
                delta_feature,
                bias=None,
                dim_key=self.dim_state_fc,
                dim_value=self.dim_state_fc,
                num_heads=8,
                dropout_rate=0.0)

            logits = []
            for k in range(self.num_modes):
                net = attn_output[:, k, :]
                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc1')
                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc2')
                net = slim.fully_connected(
                    net,
                    1,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logit')
                logits.append(net)

            mode_logit = tf.concat(logits, axis=-1, name='mode_logit')
            mode_prob = tf.nn.softmax(mode_logit, name='mode_prob')

            return mode_prob, mode_logit
