"""Latent Dynamics Model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.networks import network
from tf_agents.utils import nest_utils

from robovat.utils.logging import logger  # NOQA
from robovat.networks.layer_utils import batch_norm_params
from robovat.networks.layer_utils import two_layer_residual_block  # NOQA

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


def expand_and_tile(x, multiple, axis=0):
    n_dims = len(x.get_shape())
    multiples = axis * [1] + [multiple] + (n_dims - axis) * [1]
    return tf.tile(tf.expand_dims(x, axis), multiples)


class CircleNet(network.Network):
    """Latent Dynamics Model."""

    # def __init__(self,
    #              time_step_spec,
    #              action_spec,
    #              is_training,
    #              config=None,
    #              name='model'):
    #     """Initialize."""
    #     super(CircleNet, self).__init__(
    #         observation_spec=(),
    #         action_spec=action_spec,
    #         state_spec=(),
    #         name=name)
    #
    #     self._time_step_spec = time_step_spec
    #     self._action_spec = action_spec
    #
    #     self.is_training = is_training
    #     self.config = config
    #
    #     self.num_movables = int(
    #         time_step_spec.observation['movable'].shape[0])
    #
    #     self.dim_z = config.DIM_Z
    #     self.dim_z_fc = config.DIM_Z_FC
    #     self.dim_state_fc = config.DIM_STATE_FC
    #     self.dim_start_position_fc = config.DIM_START_POSITION_FC
    #     self.dim_action_fc = self.dim_start_position_fc
    #
    # def get_inputs_from_batch(self, batch):
    #     with tf.variable_scope('get_inputs_from_batch'):
    #         state = batch['state']
    #         action = batch['action']
    #         next_state = None
    #         return state, action, next_state
    #
    # def get_inputs_online(self, observation, action):
    #     with tf.variable_scope('get_inputs_online'):
    #         state = {
    #             'movable': observation['movable']
    #         }
    #         return state, action
    #
    # def call(self,
    #          observation,
    #          action,
    #          network_state=()):
    #     """Call the network in the policy."""
    #     with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
    #         outputs = dict()
    #
    #         state, z = self.get_inputs_online(observation, action)
    #         # state: [1, shape_state]
    #         # z: [num_samples, dim_z] or [num_samples, num_steps, dim_z]
    #
    #         sample_shape = z.get_shape()
    #         num_samples = int(sample_shape[0])
    #
    #         h = self.state_encoder(state)
    #         assert h.get_shape()[0] == 1
    #         h = tf.tile(h, [num_samples, 1], name='tiled_h')
    #         # [num_samples, dim_h]
    #
    #         ####
    #         # Generate action.
    #         ####
    #         action = self.action_decoder(h, z)
    #         outputs['action'] = action
    #
    #         outputs = self.postprocess_outputs(outputs)
    #
    #         return outputs
    #
    # def forward(self, batch):
    #     """Forward the network in the policy or on the dataset."""
    #     with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
    #         outputs = dict()
    #
    #         state, action, next_state = self.get_inputs_from_batch(batch)
    #         # state: [batch_size, shape_state]
    #         # action: [batch_size, shape_action]
    #         # next_state: [batch_size, shape_state]
    #
    #         ####
    #         # State Encoding.
    #         ####
    #         h = self.state_encoder(state)
    #         outputs['h'] = h
    #         # [batch_size, dim_h]
    #
    #         ####
    #         # Inference.
    #         ####
    #         z_mean, z_stddev = self.infer_z(h, action)
    #         z = self.sample_z(z_mean, z_stddev)
    #         outputs['z_mean'] = z_mean
    #         outputs['z_stddev'] = z_stddev
    #         outputs['z'] = z
    #         # [batch_size, dim_z]
    #
    #         ####
    #         # Prediction of action.
    #         ####
    #         action_decoded = self.action_decoder(h, z)
    #         outputs['action'] = action_decoded
    #         # [batch_size, shape_state]
    #
    #         ####
    #         # Generatation of action (Debug).
    #         ####
    #         z_gen = self.sample_z(tf.zeros_like(z_mean),
    #                               tf.ones_like(z_stddev))
    #         action_gen = self.action_decoder(
    #             h, z_gen, is_training=False)
    #         outputs['action_gen'] = action_gen
    #         # [batch_size, shape_state]
    #
    #         return outputs
    #
    # def postprocess_outputs(self, outputs):
    #     return outputs
    #
    # def sample_mode(self, mode_prior_logit):
    #     mode = tf.random.categorical(mode_prior_logit, 1)
    #     mode = tf.squeeze(mode, -1, name='mode')
    #     return mode
    #
    # def sample_z(self, mean, stddev):
    #     with tf.variable_scope('sample_z'):
    #         z = mean + stddev * tf.random_normal(
    #             tf.shape(stddev), 0., 1., dtype=tf.float32)
    #         return tf.identity(z, name='z')
    #
    # def state_encoder(self, state, is_training=None):
    #     is_training = is_training or self.is_training
    #     with tf.variable_scope('state_encoder', reuse=tf.AUTO_REUSE):
    #         return self._state_encoder(state, is_training)
    #
    # def infer_z(self, h, action, is_training=None):
    #     is_training = is_training or self.is_training
    #     with tf.variable_scope('infer_z', reuse=tf.AUTO_REUSE):
    #         z_mean, z_stddev = self._infer_z(h, action, is_training)
    #         return z_mean, z_stddev
    #
    # def action_decoder(self, h, z, is_training=None):
    #     is_training = is_training or self.is_training
    #     with tf.variable_scope('action_decoder', reuse=tf.AUTO_REUSE):
    #         return self._action_decoder(h, z, is_training)
    #
    # def _get_action_feature(self, action, is_training):
    #     with slim.arg_scope([slim.batch_norm], is_training=is_training):
    #         with slim.arg_scope(
    #                 [slim.fully_connected],
    #                 activation_fn=tf.nn.relu,
    #                 normalizer_fn=slim.batch_norm,
    #                 normalizer_params=batch_norm_params):
    #
    #             with tf.variable_scope('position_encoder'):
    #                 net = action['position']
    #                 net = slim.fully_connected(
    #                     net, self.dim_start_position_fc, scope='fc')
    #                 position_feature = net
    #
    #             action_feature = position_feature
    #
    #             return action_feature
    #
    # def _state_encoder(self, state, is_training):
    #     with slim.arg_scope([slim.batch_norm], is_training=is_training):
    #         with slim.arg_scope(
    #                 [slim.fully_connected],
    #                 activation_fn=tf.nn.relu,
    #                 normalizer_fn=slim.batch_norm,
    #                 normalizer_params=batch_norm_params):
    #
    #             position = tf.reshape(
    #                 state['movable'], [-1, self.num_movables * 2])
    #             net = position
    #             net = slim.fully_connected(
    #                 net, self.dim_state_fc, scope='fc1')
    #             net = slim.fully_connected(
    #                 net, self.dim_state_fc, scope='fc2')
    #             h = slim.fully_connected(
    #                 net,
    #                 self.dim_state_fc,
    #                 activation_fn=None,
    #                 normalizer_fn=None,
    #                 scope='h')
    #
    #             return h
    #
    # def _infer_z(self, h, action, is_training):
    #     with slim.arg_scope([slim.batch_norm], is_training=is_training):
    #         with slim.arg_scope(
    #                 [slim.fully_connected],
    #                 activation_fn=tf.nn.relu,
    #                 normalizer_fn=slim.batch_norm,
    #                 normalizer_params=batch_norm_params):
    #
    #             action_feature = self._get_action_feature(action, is_training)
    #             net = tf.concat([h, action_feature], axis=-1)
    #
    #             net = slim.fully_connected(net, self.dim_z_fc, scope='fc1')
    #             net = two_layer_residual_block(
    #                 net, self.dim_z_fc, is_training, scope='block1')
    #             net = two_layer_residual_block(
    #                 net, self.dim_z_fc, is_training, scope='block2')
    #             net = slim.fully_connected(net, self.dim_z_fc, scope='fc2')
    #
    #             gaussian_params = slim.fully_connected(
    #                 net,
    #                 2 * self.dim_z,
    #                 activation_fn=None,
    #                 normalizer_fn=None,
    #                 scope='gaussian_params')
    #             z_mean = tf.identity(
    #                 gaussian_params[:, :self.dim_z], name='z_mean')
    #             z_stddev = tf.add(
    #                 tf.nn.softplus(gaussian_params[:, self.dim_z:]), 1e-6)
    #             z_stddev = tf.identity(z_stddev, name='z_stddev')
    #
    #             return z_mean, z_stddev
    #
    # def _action_decoder(self, h, z, is_training):
    #     with slim.arg_scope([slim.batch_norm], is_training=is_training):
    #         with slim.arg_scope(
    #                 [slim.fully_connected],
    #                 activation_fn=tf.nn.relu,
    #                 normalizer_fn=slim.batch_norm,
    #                 normalizer_params=batch_norm_params):
    #             z_feature = slim.fully_connected(
    #                 z, self.dim_z_fc, scope='z_feature')
    #             net = tf.concat([h, z_feature], axis=-1)
    #
    #             net = slim.fully_connected(
    #                 net, self.dim_action_fc, scope='fc1')
    #             net = two_layer_residual_block(
    #                 net, self.dim_action_fc, is_training, scope='block1')
    #             net = two_layer_residual_block(
    #                 net, self.dim_action_fc, is_training, scope='block2')
    #             net = slim.fully_connected(
    #                 net, self.dim_action_fc, scope='fc2')
    #             action_feature = net
    #
    #             with tf.variable_scope('position_decoder'):
    #                 net = action_feature
    #                 net = slim.fully_connected(
    #                     net, self.dim_start_position_fc, scope='fc')
    #                 position = slim.fully_connected(
    #                     net,
    #                     2,
    #                     activation_fn=None,
    #                     normalizer_fn=None,
    #                     scope='position')
    #
    #             action = {
    #                 'position': position,
    #             }
    #
    #             return action


class MixtureCircleNet(network.Network):
    """Latent Dynamics Model."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(MixtureCircleNet, self).__init__(
            observation_spec=(),
            action_spec=action_spec,
            state_spec=(),
            name=name)

        self._time_step_spec = time_step_spec
        self._action_spec = action_spec

        self.is_training = is_training
        self.config = config

        self.num_movables = int(
            time_step_spec.observation['movable'].shape[0])

        self.dim_z = config.DIM_Z
        self.dim_z_fc = config.DIM_Z_FC
        self.dim_mode_fc = config.DIM_MODE_FC
        self.dim_state_fc = config.DIM_STATE_FC
        self.dim_start_position_fc = config.DIM_START_POSITION_FC
        self.dim_action_fc = self.dim_start_position_fc
        self.num_modes = config.ACTION['NUM_MODES']
        print('num modes: %d' % self.num_modes)

    def get_inputs_from_batch(self, batch):
        with tf.variable_scope('get_inputs_from_batch'):
            state = batch['state']
            action = batch['action']
            next_state = None
            return state, action, next_state

    def get_inputs_online(self, observation, action):
        with tf.variable_scope('get_inputs_online'):
            state = {
                'movable': observation['movable']
            }
            return state, action

    def call(self,
             observation,
             action,
             network_state=()):
        """Call the network in the policy."""
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            outputs = dict()

            state, z = self.get_inputs_online(observation, action)
            # state: [1, shape_state]
            # z: [num_samples, dim_z] or [num_steps, num_samples, dim_z]

            sample_shape = z.get_shape()
            num_samples = int(sample_shape[0])

            h = self.state_encoder(state)
            assert h.get_shape()[0] == 1
            h = tf.tile(h, [num_samples, 1], name='tiled_h')
            # [num_samples, dim_h]

            z = expand_and_tile(z, self.num_modes, axis=0)

            ####
            # Sample Mode
            ####
            mode_prior_logit, mode_prior = self.predict_mode(h)
            mode = self.sample_mode(mode_prior_logit)
            mode = self.sample_mode(tf.ones_like(mode_prior_logit))
            # mode = 1 * tf.ones_like(mode)  # TODO(debug)
            inds = tf.range(num_samples, dtype=tf.int64)
            inds = tf.stack([mode, inds], axis=-1)

            ####
            # Generate action.
            ####
            z = self.sample_z(tf.zeros_like(z), tf.ones_like(z))
            action = self.action_decoder(h, z)
            action = nest.map_structure(
                lambda x: tf.gather_nd(x, inds), action)
            # action['mode'] = mode
            # action['use_mode'] = tf.ones_like(mode, dtype=tf.int64)
            outputs['action'] = action 
            outputs['mode'] = mode
            outputs['mode_prior_logit'] = mode_prior_logit
            outputs['mode_prior'] = mode_prior

            outputs = self.postprocess_outputs(outputs)

            return outputs

    def forward(self, batch):
        """Forward the network in the policy or on the dataset."""
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            outputs = dict()

            state, action, next_state = self.get_inputs_from_batch(batch)
            # state: [batch_size, shape_state]
            # action: [batch_size, shape_action]
            # next_state: [batch_size, shape_state]

            ####
            # State Encoding.
            ####
            h = self.state_encoder(state)
            # [batch_size, dim_h]

            ####
            # Inference.
            ####
            mode_posterior_logit, mode_posterior = self.infer_mode(h, action)
            outputs['mode_posterior_logit'] = mode_posterior_logit
            outputs['mode_posterior'] = mode_posterior
            # [num_samples, num_modes]

            z_mean, z_stddev = self.infer_z(h, action)
            z = self.sample_z(z_mean, z_stddev)
            outputs['z_mean'] = z_mean
            outputs['z_stddev'] = z_stddev
            outputs['z'] = z
            # List of [batch_size, num_modes, dim_z]

            ####
            # Prior.
            ####
            mode_prior_logit, mode_prior = self.predict_mode(h)
            outputs['mode_prior_logit'] = mode_prior_logit
            outputs['mode_prior'] = mode_prior
            # [num_samples, num_modes]

            ####
            # Prediction of action.
            ####
            action_decoded = self.action_decoder(h, z)
            outputs['action'] = action_decoded
            # [batch_size, num_modes, shape_state]

            ####
            # Generatation of action (Debug).
            ####
            # TODO(no h)
            h = tf.zeros_like(h)

            h = tf.stop_gradient(h)

            z_gen = self.sample_z(tf.zeros_like(z_mean), tf.ones_like(z_stddev))
            action_gen = self.action_decoder(h, z_gen, is_training=False)
            outputs['action_gen'] = action_gen
            # [batch_size, num_modes, shape_state]

            dis_logit_gen = []
            for k in range(self.num_modes):
                action_gen_k = nest.map_structure(lambda x: x[k], action_gen)
                dis_logit_gen_k = self.action_discriminator(
                    h, action_gen_k)
                dis_logit_gen.append(dis_logit_gen_k)
            dis_logit_gen = tf.concat(dis_logit_gen, axis=0)

            dis_logit_real = self.action_discriminator(h, action)
            dis_logit_real = tf.tile(dis_logit_real, [self.num_modes, 1])
            outputs['dis_logit_gen'] = dis_logit_gen
            outputs['dis_logit_real'] = dis_logit_real

            gradient_penalty = 0.0
            for k in range(self.num_modes):
                action_gen_k = nest.map_structure(lambda x: x[k], action_gen)
                gradient_penalty += self.wasserstein_gradient_penalty(
                    h, action, action_gen_k)
            outputs['gradient_penalty'] = gradient_penalty

            return outputs

    def postprocess_outputs(self, outputs):
        return outputs

    def sample_mode(self, mode_prior_logit):
        mode = tf.random.categorical(mode_prior_logit, 1)
        mode = tf.squeeze(mode, -1, name='mode')
        return mode

    def sample_z(self, mean, stddev):
        with tf.variable_scope('sample_z'):
            z = mean + stddev * tf.random_normal(
                tf.shape(stddev), 0., 1., dtype=tf.float32)
            return tf.identity(z, name='z')

    def predict_mode(self, h, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_mode', reuse=tf.AUTO_REUSE):
            return self._predict_mode(h, is_training)

    def infer_mode(self, h, action, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('infer_mode', reuse=tf.AUTO_REUSE):
            return self._infer_mode(h, action, is_training)

    def infer_z(self, h, action, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('infer_z', reuse=tf.AUTO_REUSE):
            z_mean = []
            z_stddev = []

            for k in range(self.num_modes):
                with tf.variable_scope('infer_z_mode%d' % (k),
                                       reuse=tf.AUTO_REUSE):
                    z_mean_k, z_stddev_k = self._infer_z(h, action, is_training)
                    z_mean.append(z_mean_k)
                    z_stddev.append(z_stddev_k)

            z_mean = tf.stack(z_mean, axis=0, name='z_mean')
            z_stddev = tf.stack(z_stddev, axis=0, name='z_stddev')
            return z_mean, z_stddev

    def state_encoder(self, state, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('state_encoder', reuse=tf.AUTO_REUSE):
            return self._state_encoder(state, is_training)

    def action_decoder(self, h, z, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('action_decoder', reuse=tf.AUTO_REUSE):
            action = []
            for k in range(self.num_modes):
                z_k = tf.identity(z[k], name='z_%d' % k)
                with tf.variable_scope('action_decoder_mode%d' % (k),
                                       reuse=tf.AUTO_REUSE):
                    action_k = self._action_decoder(h, z_k, is_training)
                    action.append(action_k)

            action = nest_utils.stack_nested_tensors(action)
            return action

    def action_discriminator(self, h, action, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('action_discriminator', reuse=tf.AUTO_REUSE):
            return self._action_discriminator(h, action, is_training)

    def wasserstein_gradient_penalty(self, h, action_real, action_gen,
                                     target=1.0):
        batch_size = int(h.shape[0])

        alpha = tf.random_uniform(shape=[batch_size])

        interpolates = dict()
        for key in action_real.keys():
            difference = action_gen[key] - action_real[key]
            alpha_ = tf.reshape(
                alpha, [batch_size] + [1] * (difference.shape.ndims - 1))
            interpolates[key] = action_real[key] + alpha_ * difference

        with tf.variable_scope('action_discriminator', reuse=tf.AUTO_REUSE):
            disc_interpolates = self._action_discriminator(
                h, interpolates, is_training=True)

        gradients = tf.gradients(disc_interpolates, list(interpolates.values()))

        gradient_squares = 0.0
        for grad in gradients:
            gradient_squares += tf.reduce_sum(
                tf.square(grad), axis=list(range(1, grad.shape.ndims)))

        # Propagate shape information, if possible.
        if isinstance(batch_size, int):
            gradient_squares.set_shape([batch_size] +
                                       gradient_squares.shape.as_list()[1:])

        # For numerical stability, add epsilon to the sum before taking the
        # square root. Note tf.norm does not add epsilon.
        slopes = tf.sqrt(gradient_squares + 1e-10)
        penalties = slopes / target - 1.0
        penalties_squared = tf.square(penalties)
        penalty = tf.losses.compute_weighted_loss(
            penalties_squared)

        return penalty

    def _get_action_feature(self, action, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                with tf.variable_scope('position_encoder'):
                    net = action['position']
                    net = slim.fully_connected(
                        net, self.dim_start_position_fc, scope='fc')
                    position_feature = net

                action_feature = position_feature

                return action_feature

    def _predict_mode(self, h, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                net = h
                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc1')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block1')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block2')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block3')
                logit = slim.fully_connected(
                    net,
                    self.num_modes,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logit')
                prob = tf.nn.softmax(logit, name='prob')

                return logit, prob

    def _state_encoder(self, state, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                position = tf.reshape(
                    state['movable'], [-1, self.num_movables * 2])
                net = position
                net = slim.fully_connected(net, self.dim_state_fc, scope='fc1')
                net = slim.fully_connected(net, self.dim_state_fc, scope='fc2')
                h = slim.fully_connected(
                    net,
                    self.dim_state_fc,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='h')

                return h

    def _infer_mode(self, h, action, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                action_feature = self._get_action_feature(action, is_training)
                net = tf.concat([h, action_feature], axis=-1)

                # TODO(no h)
                net = action_feature

                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc1')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block1')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block2')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block3')

                logit = slim.fully_connected(
                    net,
                    self.num_modes,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logit')
                prob = tf.nn.softmax(logit, name='prob')

                return logit, prob

    def _infer_z(self, h, action, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                action_feature = self._get_action_feature(action, is_training)
                net = tf.concat([h, action_feature], axis=-1)

                # TODO(no h)
                net = action_feature

                net = slim.fully_connected(
                    net, self.dim_z_fc, scope='fc1')
                net = two_layer_residual_block(
                    net, self.dim_z_fc, is_training, scope='block1')
                net = two_layer_residual_block(
                    net, self.dim_z_fc, is_training, scope='block2')
                net = two_layer_residual_block(
                    net, self.dim_z_fc, is_training, scope='block3')

                gaussian_params = slim.fully_connected(
                    net,
                    2 * self.dim_z,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='gaussian_params')
                z_mean = tf.identity(
                    gaussian_params[:, :self.dim_z], name='z_mean')
                z_stddev = tf.add(
                    tf.nn.softplus(gaussian_params[:, self.dim_z:]), 1e-6)
                z_stddev = tf.identity(z_stddev, name='z_stddev')

                return z_mean, z_stddev

    def _action_decoder(self, h, z, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                z_feature = slim.fully_connected(
                    z, self.dim_z_fc, scope='z_feature')
                net = tf.concat([h, z_feature], axis=-1)

                # TODO(no h)
                net = z_feature

                net = slim.fully_connected(
                    net, self.dim_action_fc, scope='fc1')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block1')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block2')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block3')
                feature = net

                with tf.variable_scope('position_decoder'):
                    net = feature
                    net = slim.fully_connected(
                        net, self.dim_start_position_fc, scope='fc1')
                    position = slim.fully_connected(
                        net,
                        2,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='position')

                action = {
                    'position': position,
                }

                return action

    def _action_discriminator(self, h, action, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                action_feature = self._get_action_feature(action, is_training)
                net = tf.concat([h, action_feature], axis=-1)

                # TODO(no h)
                net = action_feature

                net = slim.fully_connected(
                    net, self.dim_action_fc, scope='fc1')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block1')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block2')
                net = two_layer_residual_block(
                    net, self.dim_action_fc, is_training, scope='block3')
                # logit = slim.fully_connected(
                #     net,
                #     1,
                #     activation_fn=None,
                #     normalizer_fn=None,
                #     scope='logit')
                logit = tf.layers.dense(
                    net,
                    1,
                    use_bias=False,
                    name='logit')

                return logit
