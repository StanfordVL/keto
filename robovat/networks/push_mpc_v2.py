"""Model predictive controller for the pushing environment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from robovat.envs.reward_fns import manip_reward
from robovat.networks import mpc
from robovat.networks.layer_utils import batch_norm_params  # NOQA
from robovat.networks.layer_utils import two_layer_residual_block  # NOQA
from robovat.utils.logging import logger  # NOQA

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


NORMALIZER_FN = slim.batch_norm
NORMALIZER_PARAMS = batch_norm_params

# NORMALIZER_FN = None
# NORMALIZER_PARAMS = None


class PushSMPC(mpc.SMPC):

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 config=None,
                 name='model'):
        """Initialize."""
        super(PushSMPC, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training,
            config=config,
            name=name)

        self.num_bodies = int(
            time_step_spec.observation['position'].shape[0])

        self.dim_z_fc = config.DIM_Z_FC
        self.dim_c_fc = config.DIM_C_FC
        self.dim_position_fc = config.DIM_POSITION_FC
        self.dim_state_fc = self.dim_position_fc
        self.dim_action_fc = config.DIM_ACTION_FC

        self.reward_fn = manip_reward.get_reward_fn(config.TASK_NAME)

        self.num_samples = config.NUM_SAMPLES

        assert len(self._action_spec['start'].shape) == 1
        assert len(self._action_spec['motion'].shape) == 1

    def get_inputs_from_batch(self, batch):
        with tf.variable_scope('get_inputs_from_batch'):
            state = batch['state']
            action = batch['action']
            next_state = batch['next_state']
            return state, action, next_state

    def get_inputs_online(self, observation):
        with tf.variable_scope('get_inputs_online'):
            state = {
                'position': observation['position'][..., :2],
            }
            return state

    def get_reward(self, state, next_state):
        if isinstance(state, dict):
            state = state['position'][..., :2]
            next_state = next_state['position'][..., :2]

        reward, termination = tf.py_func(
            self.reward_fn,
            [state, next_state],
            [tf.float32, tf.bool])

        reward = tf.reshape(reward, [self.num_samples])
        termination = tf.reshape(termination, [self.num_samples])
        return reward, termination

    def _encode_state(self, state, is_training):
        position = state['position']

        encoded_state = dict()
        for key, value in state.items():
            encoded_state[key] = value

        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=NORMALIZER_FN,
                    normalizer_params=NORMALIZER_PARAMS):

                if 'encoded_position' not in state:
                    with tf.variable_scope('encode_position'):
                        encoded_position = slim.repeat(
                            position, 2,
                            slim.fully_connected, self.dim_position_fc,
                            scope='fc')
                        encoded_state['encoded_position'] = tf.identity(
                            encoded_position, 'encoded_position')

                return encoded_state

    def _encode_action(self, action, is_training):
        start = tf.identity(action['start'], 'start')
        motion = tf.identity(action['motion'], 'motion')

        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    normalizer_fn=NORMALIZER_FN,
                    normalizer_params=NORMALIZER_PARAMS):
                net = tf.concat([start, motion], axis=-1)
                net = slim.repeat(
                    net, 2,
                    slim.fully_connected, self.dim_action_fc, scope='fc')
                return net 

    def _encode_dynamics(self, state, next_state, is_training):
        position = state['position']
        next_position = next_state['position']
        encoded_position = state['encoded_position']

        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    normalizer_fn=NORMALIZER_FN,
                    normalizer_params=NORMALIZER_PARAMS):

                net = tf.subtract(next_position, position, 'delta_position')
                net = slim.repeat(
                    net, 2,
                    slim.fully_connected, self.dim_position_fc, scope='fc')
                encoded_delta = net

                net = tf.concat([encoded_position, encoded_delta], axis=-1)
                net = slim.repeat(
                    net, 2,
                    slim.fully_connected, self.dim_position_fc,
                    scope='block1')

                net = tf.reduce_sum(net, axis=1)
                net = slim.repeat(
                    net, 2,
                    slim.fully_connected, self.dim_position_fc,
                    scope='block2')

                return net

    def _encode_c(self, c, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    normalizer_fn=NORMALIZER_FN,
                    normalizer_params=NORMALIZER_PARAMS):
                net = slim.repeat(
                    c, 2,
                    slim.fully_connected, self.dim_c_fc, scope='fc')
                return net 

    def _encode_z(self, z, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    normalizer_fn=NORMALIZER_FN,
                    normalizer_params=NORMALIZER_PARAMS):
                net = slim.repeat(
                    z, 2,
                    slim.fully_connected, self.dim_z_fc, scope='fc')
                return net

    def _infer_c(self, state, next_state, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=NORMALIZER_FN,
                    normalizer_params=NORMALIZER_PARAMS):

                with tf.variable_scope('encode_dynamics', reuse=tf.AUTO_REUSE):
                    encoded_dynamics = self._encode_dynamics(
                        state, next_state, is_training=is_training)

                net = encoded_dynamics
                gaussian_params = slim.fully_connected(
                    net,
                    2 * self.dim_c,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='gaussian_params')
                c_mean = tf.identity(
                    gaussian_params[:, :self.dim_c], name='c_mean')
                c_stddev = tf.add(
                    tf.nn.softplus(gaussian_params[:, self.dim_c:]),
                    1e-6,
                    name='c_stddev')
                return c_mean, c_stddev

    def _infer_z(self, state, action, next_state, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=NORMALIZER_FN,
                    normalizer_params=NORMALIZER_PARAMS):

                with tf.variable_scope('encode_dynamics', reuse=tf.AUTO_REUSE):
                    encoded_dynamics = self._encode_dynamics(
                        state, next_state, is_training=is_training)

                with tf.variable_scope('encode_action'):
                    encoded_action = self._encode_action(
                        action, is_training=is_training)

                net = tf.concat([encoded_dynamics, encoded_action], axis=-1)
                net = slim.repeat(
                    net, 2,
                    slim.fully_connected, self.dim_z_fc, scope='fc')

                gaussian_params = slim.fully_connected(
                    net,
                    2 * self.dim_z,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='gaussian_params')
                z_mean = tf.identity(
                    gaussian_params[:, :self.dim_z], name='z_mean')
                z_stddev = tf.add(
                    tf.nn.softplus(gaussian_params[:, self.dim_z:]),
                    1e-6,
                    name='z_stddev')
                return z_mean, z_stddev

    def _predict_action(self, state, z, next_state, is_training):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=NORMALIZER_FN,
                    normalizer_params=NORMALIZER_PARAMS):

                with tf.variable_scope('encode_dynamics', reuse=tf.AUTO_REUSE):
                    encoded_dynamics = self._encode_dynamics(
                        state, next_state, is_training=is_training)

                with tf.variable_scope('encode_z'):
                    encoded_z = slim.repeat(
                        z, 2, slim.fully_connected, self.dim_z_fc, scope='fc')

                net = tf.concat([encoded_dynamics, encoded_z], axis=-1)
                net = slim.repeat(
                    net, 2,
                    slim.fully_connected, self.dim_z_fc, scope='fc')
                start = slim.fully_connected(
                    net,
                    int(self._action_spec['start'].shape[0]),
                    activation_fn=tf.tanh,  # TODO: Is this better?
                    normalizer_fn=None,
                    scope='start')
                motion = slim.fully_connected(
                    net,
                    int(self._action_spec['motion'].shape[0]),
                    activation_fn=tf.tanh,  # TODO: Is this better?
                    normalizer_fn=None,
                    scope='motion')

                action = {
                    'start': start,
                    'motion': motion,
                }
                return action

    def _predict_transition(self, state, c, is_training):
        position = state['position']
        encoded_position = state['encoded_position']

        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=NORMALIZER_FN,
                    normalizer_params=NORMALIZER_PARAMS):

                with tf.variable_scope('encode_c', reuse=tf.AUTO_REUSE):
                    encoded_c = slim.repeat(
                        c, 2, slim.fully_connected, self.dim_c_fc, scope='fc')

                encoded_c = tf.tile(tf.expand_dims(encoded_c, 1),
                                    [1, self.num_bodies, 1],
                                    'tiled_encoded_c')

                with tf.variable_scope('pred_state'):
                    net = tf.concat([encoded_position, encoded_c], axis=-1)
                    net = slim.repeat(
                        net, 2,
                        slim.fully_connected, self.dim_position_fc, scope='fc')
                    delta_position = slim.fully_connected(
                        net,
                        2,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='delta_position')
                    pred_position = tf.add(
                        position, delta_position, name='pred_position')
                    pred_state = {
                        'position': pred_position
                    }

                return pred_state

    def _forward_dynamics(self, state, action, is_training):
        position = state['position']
        encoded_position = state['encoded_position']

        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope(
                    [slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    normalizer_fn=NORMALIZER_FN,
                    normalizer_params=NORMALIZER_PARAMS):

                with tf.variable_scope('encode_action'):
                    encoded_action = self._encode_action(
                        action, is_training=is_training)

                encoded_action = tf.tile(tf.expand_dims(encoded_action, 1),
                                         [1, self.num_bodies, 1],
                                         'tiled_encoded_action')

                with tf.variable_scope('pred_state'):
                    net = tf.concat([encoded_position, encoded_action], axis=-1)
                    net = slim.repeat(
                        net, 2,
                        slim.fully_connected, self.dim_position_fc, scope='fc')
                    delta_position = slim.fully_connected(
                        net,
                        2,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='delta_position')
                    pred_position = tf.add(
                        position, delta_position, name='pred_position')
                    pred_state = {
                        'position': pred_position
                    }

                with tf.variable_scope('valid'):
                    net = tf.concat([encoded_position, encoded_action], axis=-1)
                    net = slim.repeat(
                        net, 2,
                        slim.fully_connected, self.dim_position_fc, scope='fc')
                    net = tf.reduce_sum(net, axis=1)
                    net = slim.fully_connected(net, self.dim_state_fc,
                                               scope='final_fc')
                    valid_logit = slim.fully_connected(
                        net,
                        1,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='valid_logit')

                return pred_state, valid_logit
