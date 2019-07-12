from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from easydict import EasyDict as edict

from robovat.networks.layer_utils import batch_norm_params  # NOQA
from robovat.networks.layer_utils import two_layer_residual_block  # NOQA
from robovat.networks.pointnet import pointnet_encoder  # NOQA
from robovat.utils.logging import logger  # NOQA

nest = tf.contrib.framework.nest
slim = tf.contrib.slim


NORMALIZER_FN = slim.batch_norm
NORMALIZER_PARAMS = batch_norm_params


DELTA_POSITION_RANGE = 0.3


def encode_action(action, is_training, config):
    config = edict(config)

    start = tf.identity(action['start'], 'start')
    motion = tf.identity(action['motion'], 'motion')

    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope(
                [slim.fully_connected],
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_start'):
                net = slim.repeat(
                    start,
                    1,
                    slim.fully_connected,
                    config.dim_fc_action,
                    scope='fc')
                encoded_start = net

            with tf.variable_scope('encode_motion'):
                net = slim.repeat(
                    motion,
                    1,
                    slim.fully_connected,
                    config.dim_fc_action,
                    scope='fc')
                encoded_motion = net

            encoded_action = tf.concat(
                [encoded_start, encoded_motion], axis=-1,
                name='encoded_action')

            return encoded_action


def encode_state(state, is_training, config):
    config = edict(config)

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
                        position,
                        1,
                        slim.fully_connected,
                        config.dim_fc_state,
                        scope='fc')
                    encoded_state['encoded_position'] = tf.identity(
                        encoded_position, 'encoded_position')

            if config.use_point_cloud:
                if 'encoded_cloud' not in state:
                    cloud = state['point_cloud']
                    with tf.variable_scope('encode_cloud'):
                        offset = tf.concat(
                            [position, tf.zeros_like(position[..., 0:1])],
                            axis=-1)
                        offset = tf.expand_dims(offset, axis=-2)
                        centered_cloud = tf.subtract(
                            cloud, offset, name='centered_cloud')
                        encoded_cloud, _ = pointnet_encoder(
                            centered_cloud,
                            conv_layers=[16, 16, 16],
                            fc_layers=[],
                            dim_output=config.dim_fc_state,
                            is_training=is_training,
                            scope='pointnet')
                        encoded_state['encoded_cloud'] = tf.identity(
                            encoded_cloud, 'encoded_cloud')

                        if is_training:
                            tf.summary.histogram('cloud_x', cloud[..., 0])
                            tf.summary.histogram('cloud_y', cloud[..., 1])
                            tf.summary.histogram('cloud_z', cloud[..., 2])
                            tf.summary.histogram('cloudc_x',
                                                 centered_cloud[..., 0])
                            tf.summary.histogram('cloudc_y',
                                                 centered_cloud[..., 1])
                            tf.summary.histogram('cloudc_z',
                                                 centered_cloud[..., 2])

            if 'interaction' not in state:
                with tf.variable_scope('encode_interaction'):
                    net = tf.subtract(
                        tf.expand_dims(position, axis=1),
                        tf.expand_dims(position, axis=2),
                    )
                    net = slim.repeat(
                        net,
                        2,
                        slim.fully_connected,
                        config.dim_fc_state,
                        scope='fc')
                    mask = tf.subtract(
                        tf.ones([config.num_bodies, config.num_bodies]),
                        tf.eye(config.num_bodies)
                    )
                    mask = tf.expand_dims(
                        tf.expand_dims(mask, axis=-1),
                        axis=0)
                    net = net * mask
                    interaction = tf.reduce_sum(net, axis=1)
                    encoded_state['interaction'] = tf.identity(
                        interaction, 'interaction')

            return encoded_state


def infer_c(state, next_state, is_training, config):
    config = edict(config)

    position = state['position']
    next_position = next_state['position']
    encoded_position = state['encoded_position']

    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_delta'):
                net = tf.subtract(next_position, position, 'delta_position')
                net = slim.repeat(
                    net,
                    1,
                    slim.fully_connected,
                    config.dim_fc_state,
                    scope='fc')
                encoded_delta = net

            net = tf.concat([encoded_position, encoded_delta], axis=-1)
            net = slim.repeat(
                net,
                2,
                slim.fully_connected,
                config.dim_fc_state,
                scope='block1')

            net = tf.reduce_sum(net, axis=1)

            net = slim.repeat(
                net,
                1,
                slim.fully_connected,
                config.dim_fc_state,
                scope='block2')
            gaussian_params = slim.fully_connected(
                net,
                2 * config.dim_c,
                activation_fn=None,
                normalizer_fn=None,
                scope='gaussian_params')
            c_mean = tf.identity(
                gaussian_params[:, :config.dim_c], name='c_mean')
            c_stddev = tf.add(
                tf.nn.softplus(gaussian_params[:, config.dim_c:]),
                1e-6,
                name='c_stddev')
            return c_mean, c_stddev


def infer_z(state, action, c, is_training, config):
    config = edict(config)

    encoded_position = state['encoded_position']
    interaction = state['interaction']

    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_c', reuse=tf.AUTO_REUSE):
                encoded_c = slim.repeat(
                    c, 1, slim.fully_connected, config.dim_fc_c, scope='fc')

            with tf.variable_scope('encode_effect', reuse=tf.AUTO_REUSE):
                encoded_c = tf.tile(tf.expand_dims(encoded_c, 1),
                                    [1, config.num_bodies, 1])

                if config.use_point_cloud:
                    features = [
                        encoded_position,
                        state['encoded_cloud'],
                        interaction,
                        encoded_c,
                    ] 
                else:
                    features = [
                        encoded_position,
                        interaction,
                        encoded_c,
                    ] 

                net = tf.concat(features, axis=-1)
                net = slim.repeat(
                    net,
                    2,
                    slim.fully_connected,
                    config.dim_fc_state,
                    scope='fc')

                weight = slim.fully_connected(
                    net,
                    1,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='weight')
                weight = tf.nn.softmax(weight, axis=1)
                value = slim.fully_connected(
                    net,
                    config.dim_fc_state,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='value')
                net = value * weight

                if is_training:
                    tf.summary.histogram('weight_z', weight)

                net = tf.reduce_sum(net, axis=1)
                encoded_effect = net

            with tf.variable_scope('encode_action'):
                encoded_action = encode_action(
                    action, is_training=is_training, config=config)

            net = tf.concat([encoded_action, encoded_effect], axis=-1)
            net = slim.repeat(
                net,
                2,
                slim.fully_connected,
                config.dim_fc_z,
                scope='fc')
            gaussian_params = slim.fully_connected(
                net,
                2 * config.dim_z,
                activation_fn=None,
                normalizer_fn=None,
                scope='gaussian_params')
            z_mean = tf.identity(
                gaussian_params[:, :config.dim_z], name='z_mean')
            z_stddev = tf.add(
                tf.nn.softplus(gaussian_params[:, config.dim_z:]),
                1e-6,
                name='z_stddev')

            return z_mean, z_stddev


def predict_action(state, z, c, is_training, config):
    config = edict(config)

    encoded_position = state['encoded_position']
    interaction = state['interaction']

    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_c', reuse=tf.AUTO_REUSE):
                encoded_c = slim.repeat(
                    c, 2, slim.fully_connected, config.dim_fc_c, scope='fc')

            with tf.variable_scope('encode_effect', reuse=tf.AUTO_REUSE):
                encoded_c = tf.tile(tf.expand_dims(encoded_c, 1),
                                    [1, config.num_bodies, 1])

                if config.use_point_cloud:
                    features = [
                        encoded_position,
                        state['encoded_cloud'],
                        interaction,
                        encoded_c,
                    ] 
                else:
                    features = [
                        encoded_position,
                        interaction,
                        encoded_c,
                    ] 

                net = tf.concat(features, axis=-1)
                net = slim.repeat(
                    net,
                    2,
                    slim.fully_connected,
                    config.dim_fc_state,
                    scope='fc')

                weight = slim.fully_connected(
                    net,
                    1,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='weight')
                weight = tf.nn.softmax(weight, axis=1)
                value = slim.fully_connected(
                    net,
                    config.dim_fc_state,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='value')
                net = value * weight

                if is_training:
                    tf.summary.histogram('weight_action', weight)

                net = tf.reduce_sum(net, axis=1)
                encoded_effect = net

            with tf.variable_scope('transform_z'):
                net = slim.fully_connected(
                    encoded_effect, config.dim_fc_z, scope='fc')
                gaussian_params = slim.fully_connected(
                    net,
                    2 * config.dim_z,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='gaussian_params')
                zc_mean = tf.identity(
                    gaussian_params[:, :config.dim_z], name='zc_mean')
                zc_stddev = tf.add(
                    tf.nn.softplus(gaussian_params[:, config.dim_z:]),
                    1e-6,
                    name='zc_stddev')
                z = zc_mean + zc_stddev * z

            dim_z_half = int(config.dim_z / 2)
            with tf.variable_scope('decode_start'):
                net = slim.repeat(
                    z[..., :dim_z_half],
                    2,
                    slim.fully_connected,
                    config.dim_fc_action,
                    scope='fc')
                start = slim.fully_connected(
                    net,
                    config.dim_start,
                    # int(_action_spec['start'].shape[0]),
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='start')
            with tf.variable_scope('decode_motion'):
                net = slim.repeat(
                    z[..., dim_z_half:],
                    2,
                    slim.fully_connected,
                    config.dim_fc_action,
                    scope='fc')
                motion = slim.fully_connected(
                    net,
                    config.dim_motion,
                    # int(_action_spec['motion'].shape[0]),
                    activation_fn=tf.tanh,
                    normalizer_fn=None,
                    scope='motion')

            action = {
                'start': start,
                'motion': motion,
            }
            return action


def predict_transition(state, c, is_training, config):
    config = edict(config)

    position = state['position']
    encoded_position = state['encoded_position']
    interaction = state['interaction']

    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_c', reuse=tf.AUTO_REUSE):
                encoded_c = slim.repeat(
                    c, 2, slim.fully_connected, config.dim_fc_c, scope='fc')

            encoded_c = tf.tile(tf.expand_dims(encoded_c, 1),
                                [1, config.num_bodies, 1])

            if config.use_point_cloud:
                features = [
                    encoded_position,
                    state['encoded_cloud'],
                    interaction,
                    encoded_c,
                ] 
            else:
                features = [
                    encoded_position,
                    interaction,
                    encoded_c,
                ] 

            net = tf.concat(features, axis=-1)
            net = slim.repeat(
                net,
                2,
                slim.fully_connected,
                config.dim_fc_state,
                scope='fc')
            encoded_effect = net

            with tf.variable_scope('pred_position'):
                net = slim.fully_connected(
                    encoded_effect, config.dim_fc_state, scope='fc')
                delta_position = slim.fully_connected(
                    net,
                    2,
                    activation_fn=tf.tanh,
                    normalizer_fn=None,
                    scope='delta_position')
                gate = slim.fully_connected(
                    net,
                    1,
                    activation_fn=tf.sigmoid,
                    normalizer_fn=None,
                    scope='gate')
                delta_position = delta_position * gate * DELTA_POSITION_RANGE
                pred_position = tf.add(
                    position, delta_position, name='pred_position')

                if is_training:
                    tf.summary.histogram('gate_transition', gate)

            pred_state = {
                'position': pred_position,
            }

            if config.use_point_cloud:
                with tf.variable_scope('pred_cloud'):
                    net = slim.fully_connected(
                        encoded_effect, config.dim_fc_state, scope='fc')
                    delta_cloud = slim.fully_connected(
                        net,
                        config.dim_fc_state,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='delta_cloud')
                    pred_cloud = tf.add(
                        state['encoded_cloud'], delta_cloud, name='pred_cloud')
                    pred_state['encoded_cloud'] = pred_cloud

            return pred_state


def forward_dynamics(state, action, is_training, config):
    config = edict(config)

    position = state['position']
    encoded_position = state['encoded_position']
    interaction = state['interaction']

    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_action'):
                encoded_action = encode_action(
                    action, is_training=is_training, config=config)

            encoded_action = tf.tile(tf.expand_dims(encoded_action, 1),
                                     [1, config.num_bodies, 1])

            if config.use_point_cloud:
                features = [
                    encoded_position,
                    state['encoded_cloud'],
                    interaction,
                    encoded_action,
                ] 
            else:
                features = [
                    encoded_position,
                    interaction,
                    encoded_action,
                ] 

            net = tf.concat(features, axis=-1)
            net = slim.repeat(
                net,
                2,
                slim.fully_connected,
                config.dim_fc_state,
                scope='fc')
            encoded_effect = net

            with tf.variable_scope('pred_position'):
                net = slim.fully_connected(
                    encoded_effect, config.dim_fc_state, scope='fc')
                delta_position = slim.fully_connected(
                    net,
                    2,
                    activation_fn=tf.tanh,
                    normalizer_fn=None,
                    scope='delta_position')
                gate = slim.fully_connected(
                    net,
                    1,
                    activation_fn=tf.sigmoid,
                    normalizer_fn=None,
                    scope='gate')
                delta_position = delta_position * gate * DELTA_POSITION_RANGE
                pred_position = tf.add(
                    position, delta_position, name='pred_position')

                if is_training:
                    tf.summary.histogram('gate_dynamics', gate)

            pred_state = {
                'position': pred_position,
            }

            if config.use_point_cloud:
                with tf.variable_scope('pred_cloud'):
                    net = slim.fully_connected(
                        encoded_effect, config.dim_fc_state, scope='fc')
                    delta_cloud = slim.fully_connected(
                        net,
                        config.dim_fc_state,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='delta_cloud')
                    pred_cloud = tf.add(
                        state['encoded_cloud'], delta_cloud, name='pred_cloud')
                    pred_state['encoded_cloud'] = pred_cloud

            with tf.variable_scope('valid'):
                net = encoded_effect
                net = tf.reduce_sum(net, axis=1)
                net = slim.fully_connected(net, 32, scope='fc')
                valid_logit = slim.fully_connected(
                    net,
                    1,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='valid_logit')

            return pred_state, valid_logit


################
# For baselines.
################


def infer_z_given_sa(state, action, is_training, config):
    config = edict(config)

    encoded_position = state['encoded_position']
    interaction = state['interaction']

    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_effect', reuse=tf.AUTO_REUSE):
                if config.use_point_cloud:
                    features = [
                        encoded_position,
                        state['encoded_cloud'],
                        interaction,
                    ] 
                else:
                    features = [
                        encoded_position,
                        interaction,
                    ] 

                net = tf.concat(features, axis=-1)
                net = slim.repeat(
                    net,
                    2,
                    slim.fully_connected,
                    config.dim_fc_state,
                    scope='fc')

                weight = slim.fully_connected(
                    net,
                    1,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='weight')
                weight = tf.nn.softmax(weight, axis=1)
                value = slim.fully_connected(
                    net,
                    config.dim_fc_state,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='value')
                net = value * weight

                if is_training:
                    tf.summary.histogram('weight_z', weight)

                net = tf.reduce_sum(net, axis=1)
                encoded_effect = net

            with tf.variable_scope('encode_action'):
                encoded_action = encode_action(
                    action, is_training=is_training, config=config)

            net = tf.concat([encoded_action, encoded_effect], axis=-1)
            net = slim.repeat(
                net,
                2,
                slim.fully_connected,
                config.dim_fc_z,
                scope='fc')
            gaussian_params = slim.fully_connected(
                net,
                2 * config.dim_z,
                activation_fn=None,
                normalizer_fn=None,
                scope='gaussian_params')
            z_mean = tf.identity(
                gaussian_params[:, :config.dim_z], name='z_mean')
            z_stddev = tf.add(
                tf.nn.softplus(gaussian_params[:, config.dim_z:]),
                1e-6,
                name='z_stddev')

            return z_mean, z_stddev


def predict_action_given_z(state, z, is_training, config):
    config = edict(config)

    encoded_position = state['encoded_position']
    interaction = state['interaction']

    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_effect', reuse=tf.AUTO_REUSE):

                if config.use_point_cloud:
                    features = [
                        encoded_position,
                        state['encoded_cloud'],
                        interaction,
                    ] 
                else:
                    features = [
                        encoded_position,
                        interaction,
                    ] 

                net = tf.concat(features, axis=-1)
                net = slim.repeat(
                    net,
                    2,
                    slim.fully_connected,
                    config.dim_fc_state,
                    scope='fc')

                weight = slim.fully_connected(
                    net,
                    1,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='weight')
                weight = tf.nn.softmax(weight, axis=1)
                value = slim.fully_connected(
                    net,
                    config.dim_fc_state,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='value')
                net = value * weight

                if is_training:
                    tf.summary.histogram('weight_action', weight)

                net = tf.reduce_sum(net, axis=1)
                encoded_effect = net

            with tf.variable_scope('transform_z'):
                net = slim.fully_connected(
                    encoded_effect, config.dim_fc_z, scope='fc')
                gaussian_params = slim.fully_connected(
                    net,
                    2 * config.dim_z,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='gaussian_params')
                zc_mean = tf.identity(
                    gaussian_params[:, :config.dim_z], name='zc_mean')
                zc_stddev = tf.add(
                    tf.nn.softplus(gaussian_params[:, config.dim_z:]),
                    1e-6,
                    name='zc_stddev')
                z = zc_mean + zc_stddev * z

            dim_z_half = int(config.dim_z / 2)
            with tf.variable_scope('decode_start'):
                net = slim.repeat(
                    z[..., :dim_z_half],
                    2,
                    slim.fully_connected,
                    config.dim_fc_action,
                    scope='fc')
                start = slim.fully_connected(
                    net,
                    config.dim_start,
                    # int(_action_spec['start'].shape[0]),
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='start')
            with tf.variable_scope('decode_motion'):
                net = slim.repeat(
                    z[..., dim_z_half:],
                    2,
                    slim.fully_connected,
                    config.dim_fc_action,
                    scope='fc')
                motion = slim.fully_connected(
                    net,
                    config.dim_motion,
                    # int(_action_spec['motion'].shape[0]),
                    activation_fn=tf.tanh,
                    normalizer_fn=None,
                    scope='motion')

            action = {
                'start': start,
                'motion': motion,
            }
            return action


def predict_action_concat(state, z, c, is_training, config):
    config = edict(config)

    encoded_position = state['encoded_position']
    interaction = state['interaction']

    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_c', reuse=tf.AUTO_REUSE):
                encoded_c = slim.repeat(
                    c, 2, slim.fully_connected, config.dim_fc_c, scope='fc')

            with tf.variable_scope('encode_effect', reuse=tf.AUTO_REUSE):
                encoded_c = tf.tile(tf.expand_dims(encoded_c, 1),
                                    [1, config.num_bodies, 1])

                if config.use_point_cloud:
                    features = [
                        encoded_position,
                        state['encoded_cloud'],
                        interaction,
                        encoded_c,
                    ] 
                else:
                    features = [
                        encoded_position,
                        interaction,
                        encoded_c,
                    ] 

                net = tf.concat(features, axis=-1)
                net = slim.repeat(
                    net,
                    2,
                    slim.fully_connected,
                    config.dim_fc_state,
                    scope='fc')

                weight = slim.fully_connected(
                    net,
                    1,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='weight')
                weight = tf.nn.softmax(weight, axis=1)
                value = slim.fully_connected(
                    net,
                    config.dim_fc_state,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='value')
                net = value * weight

                if is_training:
                    tf.summary.histogram('weight_action', weight)

                net = tf.reduce_sum(net, axis=1)
                encoded_effect = net

            with tf.variable_scope('encode_c', reuse=tf.AUTO_REUSE):
                encoded_z = slim.repeat(
                    z, 1, slim.fully_connected, config.dim_fc_z, scope='fc')

            with tf.variable_scope('transform_z'):
                net = tf.concat([encoded_effect, encoded_z], axis=-1)
                net = slim.fully_connected(
                    net, config.dim_fc_action, scope='fc')
                encoded_feature = net

            with tf.variable_scope('decode_start'):
                net = slim.fully_connected(
                    encoded_feature,
                    config.dim_fc_action,
                    scope='fc')
                start = slim.fully_connected(
                    net,
                    config.dim_start,
                    # int(_action_spec['start'].shape[0]),
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='start')
            with tf.variable_scope('decode_motion'):
                net = slim.fully_connected(
                    encoded_feature,
                    config.dim_fc_action,
                    scope='fc')
                motion = slim.fully_connected(
                    net,
                    config.dim_motion,
                    # int(_action_spec['motion'].shape[0]),
                    activation_fn=tf.tanh,
                    normalizer_fn=None,
                    scope='motion')

            action = {
                'start': start,
                'motion': motion,
            }
            return action
