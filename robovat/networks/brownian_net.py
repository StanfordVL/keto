"""Latent Dynamics Model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.networks import network

from robovat.utils import nest_utils
from robovat.utils.logging import logger  # NOQA
from robovat.networks.layer_utils import batch_norm_params
from robovat.networks.layer_utils import two_layer_residual_block  # NOQA

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


class BrownianNet(network.Network):
    """Latent Dynamics Model."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(BrownianNet, self).__init__(
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
        self.dim_state_fc = config.DIM_STATE_FC

    def get_inputs_from_batch(self, batch):
        with tf.variable_scope('get_inputs_from_batch'):
            state = batch['state']
            action = batch['action']
            next_state = batch['next_state']
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
            # z: [num_samples, dim_z] or [num_samples, num_steps, dim_z]

            sample_shape = z.get_shape()
            num_samples = int(sample_shape[0])

            h = self.state_encoder(state)
            assert h.get_shape()[0] == 1
            h = tf.tile(h, [num_samples, 1], name='tiled_h')
            # [num_samples, dim_h]

            ####
            # Generate action.
            ####
            pred_h = self.predict_dynamics(h, z)
            next_state = self.state_decoder(pred_h)
            outputs['next_state'] = next_state
            # [batch_size, dim_state_fc]

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
            next_h = self.state_encoder(next_state)
            # [batch_size, dim_h]

            ####
            # Inference.
            ####
            z_mean, z_stddev = self.infer_z(h, next_h)
            z = self.sample_z(z_mean, z_stddev)
            outputs['z_mean'] = z_mean
            outputs['z_stddev'] = z_stddev
            outputs['z'] = z
            # [batch_size, dim_z]

            ####
            # Prediction of state.
            ####
            pred_h = self.predict_dynamics(h, z)
            next_state = self.state_decoder(pred_h)
            outputs['next_state'] = next_state
            # [batch_size, dim_state_fc]

            ####
            # Sampling of state (Debug).
            ####
            z_samp = self.sample_z(tf.zeros_like(z_mean),
                                   tf.ones_like(z_stddev))
            pred_h_samp = self.predict_dynamics(h, z_samp, is_training=False)
            next_state_samp = self.state_decoder(pred_h_samp, is_training=False)
            outputs['next_state_samp'] = next_state_samp
            # [batch_size, shape_state]

            return outputs

    def postprocess_outputs(self, outputs):
        return outputs

    def sample_z(self, mean, stddev):
        with tf.variable_scope('sample_z'):
            z = mean + stddev * tf.random_normal(
                tf.shape(stddev), 0., 1., dtype=tf.float32)
            return tf.identity(z, name='z')

    def infer_z(self, h, next_h, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('infer_z', reuse=tf.AUTO_REUSE):
            z_mean, z_stddev = self._infer_z(h, next_h, is_training)
            return z_mean, z_stddev

    def state_encoder(self, state, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('state_encoder', reuse=tf.AUTO_REUSE):
            return self._state_encoder(state, is_training)

    def state_decoder(self, h, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('state_decoder', reuse=tf.AUTO_REUSE):
            return self._state_decoder(h, is_training)

    def predict_dynamics(self, h, z, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_dynamics', reuse=tf.AUTO_REUSE):
            return self._predict_dynamics(h, z, is_training)

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

    def _state_decoder(self, h, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                net = h
                net = slim.fully_connected(
                    net, self.dim_state_fc, scope='fc1')
                net = slim.fully_connected(
                    net, self.dim_state_fc, scope='fc2')
                movable = slim.fully_connected(
                    net,
                    self.num_movables * 2,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='movable')
                movable = tf.reshape(movable, [-1, self.num_movables, 2])

                state = {
                    'movable': movable,
                }

                return state

    def _predict_dynamics(self, h, z, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                z_feature = slim.fully_connected(
                    z, self.dim_z_fc, scope='z_feature')
                net = tf.concat([h, z_feature], axis=-1)

                net = slim.fully_connected(
                    net, self.dim_state_fc, scope='fc1')
                net = slim.fully_connected(
                    net, self.dim_state_fc, scope='fc2')

                pred_h = tf.add(h, net, name='pred_h')
                return pred_h


class MixtureBrownianNet(network.Network):
    """Latent Dynamics Model."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(MixtureBrownianNet, self).__init__(
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
            next_state = batch['next_state']
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
            # z: [num_samples, dim_z] or [num_samples, num_steps, dim_z]

            sample_shape = z.get_shape()
            num_samples = int(sample_shape[0])

            h = self.state_encoder(state, is_training=False)
            assert h.get_shape()[0] == 1
            h = tf.tile(h, [num_samples, 1], name='tiled_h')
            # [num_samples, dim_h]

            z = tf.tile(tf.expand_dims(z, 1), [1, self.num_modes, 1])

            ####
            # Sample Mode
            ####
            mode_prior, mode_prior_logit = self.predict_mode(
                h, is_training=False)
            mode = self.sample_mode(mode_prior_logit)
            # mode = self.sample_mode(tf.ones_like(mode_prior_logit))
            mode = 1 * tf.ones_like(mode)  # TODO(debug)
            inds = tf.range(num_samples, dtype=tf.int64)
            inds = tf.stack([inds, mode], axis=-1)

            ####
            # Generate action.
            ####
            # TODO
            z = self.sample_z(tf.zeros_like(z), tf.ones_like(z))
            action = self.action_decoder(h, z)
            action = nest.map_structure(
                lambda x: tf.gather_nd(x, inds), action)
            # action['mode'] = mode
            # action['use_mode'] = tf.ones_like(mode, dtype=tf.int64)
            outputs['action'] = action 
            outputs['mode'] = mode
            outputs['mode_prior'] = mode_prior
            outputs['mode_prior_logit'] = mode_prior_logit

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
            next_h = self.state_encoder(next_state)
            # [batch_size, dim_h]

            ####
            # Inference.
            ####
            mode_posterior, mode_posterior_logit = self.infer_mode(
                h, next_h, is_training=self.is_training)
            outputs['mode_posterior'] = mode_posterior
            outputs['mode_posterior_logit'] = mode_posterior_logit
            # [num_samples, num_modes]

            z_mean, z_stddev = self.infer_z(
                h, next_h, is_training=self.is_training)
            z = self.sample_z(z_mean, z_stddev)
            outputs['z_mean'] = z_mean
            outputs['z_stddev'] = z_stddev
            outputs['z'] = z
            # List of [batch_size, num_modes, dim_z]

            ####
            # Prior.
            ####
            mode_prior, mode_prior_logit = self.predict_mode(
                h, is_training=self.is_training)
            outputs['mode_prior'] = mode_prior
            outputs['mode_prior_logit'] = mode_prior_logit
            # [num_samples, num_modes]

            ####
            # Prediction of state.
            ####
            pred_h = self.predict_dynamics(h, z)
            next_state = []
            for k in range(self.num_modes):
                pred_h_k = tf.identity(pred_h[:, k], name='pred_h_%d' % k)
                next_state_k = self.state_decoder(pred_h_k)
                next_state.append(next_state_k)
            next_state = nest_utils.stack_structure(next_state)
            outputs['next_state'] = next_state
            # List of [batch_size, dim_state_fc]

            ####
            # Sampling of state (Debug).
            ####
            z_samp = self.sample_z(tf.zeros_like(z_mean),
                                   tf.ones_like(z_stddev))
            pred_h = self.predict_dynamics(h, z_samp, is_training=False)
            next_state = []
            for k in range(self.num_modes):
                pred_h_k = tf.identity(pred_h[:, k], name='pred_h_%d' % k)
                next_state_k = self.state_decoder(pred_h_k, is_training=False)
                next_state.append(next_state_k)
            next_state = nest_utils.stack_structure(next_state)
            outputs['next_state_samp'] = next_state
            # [batch_size, num_modes, shape_state]

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

    def infer_mode(self, h, next_h, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('infer_mode', reuse=tf.AUTO_REUSE):
            return self._infer_mode(h, next_h, is_training)

    def infer_z(self, h, next_h, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('infer_z', reuse=tf.AUTO_REUSE):
            z_mean = []
            z_stddev = []

            for k in range(self.num_modes):
                with tf.variable_scope('infer_z_mode%d' % (k),
                                       reuse=tf.AUTO_REUSE):
                    z_mean_k, z_stddev_k = self._infer_z(h, next_h, is_training)
                    z_mean.append(z_mean_k)
                    z_stddev.append(z_stddev_k)

            z_mean = tf.stack(z_mean, axis=1, name='z_mean')
            z_stddev = tf.stack(z_stddev, axis=1, name='z_stddev')
            return z_mean, z_stddev

    def state_encoder(self, state, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('state_encoder', reuse=tf.AUTO_REUSE):
            return self._state_encoder(state, is_training)

    def state_decoder(self, h, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('state_decoder', reuse=tf.AUTO_REUSE):
            return self._state_decoder(h, is_training)

    def predict_dynamics(self, h, z, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_dynamics', reuse=tf.AUTO_REUSE):
            pred_h = []
            for k in range(self.num_modes):
                z_k = tf.identity(z[:, k, :], name='z_%d' % k)
                with tf.variable_scope('predict_dynamics_mode%d' % (k),
                                       reuse=tf.AUTO_REUSE):
                    pred_h_k = self._predict_dynamics(h, z_k, is_training)
                    pred_h.append(pred_h_k)

            pred_h = tf.stack(pred_h, axis=1)
            return pred_h

    def _predict_mode(self, h, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                net = h
                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc1')
                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc2')
                logit = slim.fully_connected(
                    net,
                    self.num_modes,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logit')
                prob = tf.nn.softmax(logit, name='prob')

                return prob, logit

    def _infer_mode(self, h, next_h, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                net = tf.identity(next_h - h, name='delta_h')
                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc1')
                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc2')

                logit = slim.fully_connected(
                    net,
                    self.num_modes,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logit')
                prob = tf.nn.softmax(logit, name='prob')

                return prob, logit

    def _infer_z(self, h, next_h, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                net = tf.identity(next_h - h, name='delta_h')
                net = slim.fully_connected(
                    net, self.dim_z_fc, scope='fc1')
                net = slim.fully_connected(
                    net, self.dim_z_fc, scope='fc2')

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

    def _state_decoder(self, h, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                net = h
                net = slim.fully_connected(
                    net, self.dim_state_fc, scope='fc1')
                net = slim.fully_connected(
                    net, self.dim_state_fc, scope='fc2')
                movable = slim.fully_connected(
                    net,
                    self.num_movables * 2,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='movable')
                movable = tf.reshape(movable, [-1, self.num_movables, 2])

                state = {
                    'movable': movable,
                }

                return state

    def _predict_dynamics(self, h, z, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                z_feature = slim.fully_connected(
                    z, self.dim_z_fc, scope='z_feature')
                net = tf.concat([h, z_feature], axis=-1)

                net = slim.fully_connected(
                    net, self.dim_state_fc, scope='fc1')
                net = slim.fully_connected(
                    net, self.dim_state_fc, scope='fc2')

                pred_h = tf.add(h, net, name='pred_h')
                return pred_h


class MixtureBrownianNetV2(network.Network):
    """Latent Dynamics Model."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(MixtureBrownianNetV2, self).__init__(
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
            next_state = batch['next_state']
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
            # z: [num_samples, dim_z] or [num_samples, num_steps, dim_z]

            sample_shape = z.get_shape()
            num_samples = int(sample_shape[0])

            z = tf.tile(tf.expand_dims(z, 1), [1, self.num_modes, 1])

            ####
            # Sample Mode
            ####
            mode_prior, mode_prior_logit = self.predict_mode(state)
            mode = self.sample_mode(mode_prior_logit)
            # mode = self.sample_mode(tf.ones_like(mode_prior_logit))
            # mode = 0 * tf.ones_like(mode)  # TODO(debug)

            mode = tf.tile(mode, [num_samples])

            inds = tf.range(num_samples, dtype=tf.int64)
            inds = tf.stack([inds, mode], axis=-1)

            ####
            # Prediction of state.
            ####
            next_state = self.predict_dynamics(state, z)
            next_state = nest.map_structure(
                lambda x: tf.gather_nd(x, inds), next_state)
            outputs['next_state'] = next_state
            # List of [batch_size, dim_state_fc]

            outputs['mode'] = mode
            outputs['mode_prior'] = mode_prior
            outputs['mode_prior_logit'] = mode_prior_logit

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
            # Inference.
            ####
            mode_posterior, mode_posterior_logit = self.infer_mode(
                state, next_state)
            outputs['mode_posterior'] = mode_posterior
            outputs['mode_posterior_logit'] = mode_posterior_logit
            # [num_samples, num_modes]

            z_mean, z_stddev = self.infer_z(
                state, next_state)
            z = self.sample_z(z_mean, z_stddev)
            outputs['z_mean'] = z_mean
            outputs['z_stddev'] = z_stddev
            outputs['z'] = z
            # List of [batch_size, num_modes, dim_z]

            ####
            # Prior.
            ####
            mode_prior, mode_prior_logit = self.predict_mode(state)
            outputs['mode_prior'] = mode_prior
            outputs['mode_prior_logit'] = mode_prior_logit
            # [num_samples, num_modes]

            ####
            # Prediction of state.
            ####
            next_state = self.predict_dynamics(state, z)
            outputs['next_state'] = next_state
            # List of [batch_size, dim_state_fc]

            ####
            # Sampling of state (Debug).
            ####
            z_samp = self.sample_z(tf.zeros_like(z_mean),
                                   tf.ones_like(z_stddev))
            next_state = self.predict_dynamics(state, z_samp, is_training=False)
            outputs['next_state_samp'] = next_state
            # [batch_size, num_modes, shape_state]

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

    def predict_mode(self, state, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_mode', reuse=tf.AUTO_REUSE):
            return self._predict_mode(state, is_training)

    def infer_mode(self, state, next_h, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('infer_mode', reuse=tf.AUTO_REUSE):
            return self._infer_mode(state, next_h, is_training)

    def infer_z(self, state, next_state, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('infer_z', reuse=tf.AUTO_REUSE):
            z_mean = []
            z_stddev = []

            for k in range(self.num_modes):
                with tf.variable_scope('infer_z_mode%d' % (k),
                                       reuse=tf.AUTO_REUSE):
                    z_mean_k, z_stddev_k = self._infer_z(
                        state, next_state, is_training)
                    z_mean.append(z_mean_k)
                    z_stddev.append(z_stddev_k)

            z_mean = tf.stack(z_mean, axis=1, name='z_mean')
            z_stddev = tf.stack(z_stddev, axis=1, name='z_stddev')
            return z_mean, z_stddev

    def state_encoder(self, state, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('state_encoder', reuse=tf.AUTO_REUSE):
            return self._state_encoder(state, is_training)

    def predict_dynamics(self, state, z, is_training=None):
        is_training = is_training or self.is_training
        with tf.variable_scope('predict_dynamics', reuse=tf.AUTO_REUSE):
            next_state = []
            for k in range(self.num_modes):
                z_k = tf.identity(z[:, k, :], name='z_%d' % k)
                with tf.variable_scope('predict_dynamics_mode%d' % (k),
                                       reuse=tf.AUTO_REUSE):
                    next_state_k = self._predict_dynamics(
                        state, z_k, is_training)
                    next_state.append(next_state_k)

            next_state = nest_utils.stack_structure(next_state)
            return next_state

    def _get_state_feature(self, state, is_training):
        with tf.variable_scope('get_state_feature'):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                with slim.arg_scope(
                        [slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):

                    movable = tf.reshape(
                        state['movable'], [-1, self.num_movables * 2])
                    net = movable
                    net = slim.fully_connected(
                        net, self.dim_state_fc, scope='fc1')
                    net = slim.fully_connected(
                        net, self.dim_state_fc, scope='fc2')
                    h = slim.fully_connected(
                        net,
                        self.dim_state_fc,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='h')

                    return h

    def _get_dynamics_feature(self, state, next_state, is_training):
        with tf.variable_scope('get_dynamics_feature'):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                with slim.arg_scope(
                        [slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):

                    movable = tf.reshape(
                        state['movable'], [-1, self.num_movables * 2])
                    next_movable = tf.reshape(
                        next_state['movable'], [-1, self.num_movables * 2])
                    net = next_movable - movable
                    net = slim.fully_connected(
                        net, self.dim_state_fc, scope='fc1')
                    net = slim.fully_connected(
                        net, self.dim_state_fc, scope='fc2')
                    h = slim.fully_connected(
                        net,
                        self.dim_state_fc,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='h')

                    return h

    def _predict_mode(self, state, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                net = self._get_state_feature(state, is_training)
                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc1')
                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc2')
                logit = slim.fully_connected(
                    net,
                    self.num_modes,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logit')
                prob = tf.nn.softmax(logit, name='prob')

                return prob, logit

    def _infer_mode(self, state, next_state, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                net = self._get_dynamics_feature(state, next_state, is_training)
                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc1')
                net = slim.fully_connected(net, self.dim_mode_fc, scope='fc2')

                logit = slim.fully_connected(
                    net,
                    self.num_modes,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='logit')
                prob = tf.nn.softmax(logit, name='prob')

                return prob, logit

    def _infer_z(self, state, next_state, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                net = self._get_dynamics_feature(state, next_state, is_training)
                net = slim.fully_connected(
                    net, self.dim_z_fc, scope='fc1')
                net = slim.fully_connected(
                    net, self.dim_z_fc, scope='fc2')

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

    def _predict_dynamics(self, state, z, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params):

                z_feature = slim.fully_connected(
                    z, self.dim_z_fc, scope='z_feature')
                net = z_feature

                net = slim.fully_connected(
                    net, self.dim_state_fc, scope='fc1')
                net = slim.fully_connected(
                    net, self.dim_state_fc, scope='fc2')
                net = slim.fully_connected(
                    net,
                    self.num_movables * 2,
                    activation_fn=None,
                    normalizer_fn=None)
                net = tf.reshape(net, [-1, self.num_movables, 2])

                next_movable = state['movable'] + net
                next_state = {
                    'movable': next_movable
                }
                return next_state
