"""Multi-stage task problems.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np  # NOQA
import tensorflow as tf

from robovat.problems import problem
from robovat.problems import mpc_problem
from robovat.problems import baseline_problem
from robovat.utils import loss_utils

slim = tf.contrib.slim
nest = tf.contrib.framework.nest
Reduction = tf.losses.Reduction


POSITION_SCALE = 5.


class PushProblem(problem.Problem):
    """Single-stage prediction Problem."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 is_training,
                 dim_c=3):
        """Initialize."""
        super(PushProblem, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            is_training=is_training)

        # Input tensor spec.
        _position_spec = time_step_spec.observation['position']
        _point_cloud_spec = time_step_spec.observation['point_cloud']
        _start_spec = action_spec['start']
        _motion_spec = action_spec['motion']
        spec_list = [
            tf.TensorSpec(_position_spec.shape, _position_spec.dtype,
                          'position'),
            tf.TensorSpec(_position_spec.shape, _position_spec.dtype,
                          'next_position'),
            tf.TensorSpec(_point_cloud_spec.shape, _point_cloud_spec.dtype,
                          'point_cloud'),
            tf.TensorSpec(_point_cloud_spec.shape, _point_cloud_spec.dtype,
                          'next_point_cloud'),

            tf.TensorSpec(_start_spec.shape, _start_spec.dtype, 'start'),
            tf.TensorSpec(_motion_spec.shape, _motion_spec.dtype, 'motion'),

            tf.TensorSpec([1], tf.int64, 'is_safe'),
            tf.TensorSpec([1], tf.int64, 'is_effective'),
        ]
        self._spec = OrderedDict(
            [(spec.name, spec) for spec in spec_list])

        self.num_objects = int(_position_spec.shape[0])
        self.num_points = int(_position_spec.shape[1])
        self.dim_c = dim_c

    def convert_trajectory(self, trajectory):
        """Convert trajectory."""
        observation = trajectory.observation
        action = trajectory.action
        num_steps = len(trajectory.reward)

        data_list = []
        for i in range(num_steps - 1):
            data = OrderedDict([
                ('position', observation['position'][i]),
                ('next_position', observation['position'][i + 1]),
                ('point_cloud', observation['point_cloud'][i]),
                ('next_point_cloud', observation['point_cloud'][i + 1]),

                ('start', action['start'][i]),
                ('motion', action['motion'][i]),

                ('is_safe', observation['is_safe'][i + 1]),
                ('is_effective', observation['is_effective'][i + 1]),
            ])

            data_list.append(data)

        return data_list

    def preprocess(self,
                   batch,
                   augment_position=False,
                   ):
        """Augment batched point cloud."""
        with tf.variable_scope('preprocess'):

            if augment_position:
                noise_scale = 0.02
                position_shape = batch['position'].shape
                position_noise = tf.random.uniform(
                    position_shape,
                    minval=-noise_scale,
                    maxval=noise_scale,
                    name='position_noise')
                position_noise = position_noise[..., :2]
            else:
                position_noise = 0.0

            with tf.variable_scope('state'):
                position = batch['position'][..., :2]
                position += position_noise
                point_cloud = batch['point_cloud']
                state = {
                    'position': tf.identity(position, 'position'),
                    'point_cloud': tf.identity(point_cloud, 'point_cloud'),
                }

            with tf.variable_scope('next_state'):
                position = batch['next_position'][..., :2]
                position += position_noise
                point_cloud = batch['next_point_cloud']
                next_state = {
                    'position': tf.identity(position, 'position'),
                    'point_cloud': tf.identity(point_cloud, 'point_cloud'),
                }

            with tf.variable_scope('action'):
                action = {
                    'start': tf.identity(batch['start'], 'start'),
                    'motion': tf.identity(batch['motion'], 'motion'),
                }

            valid = tf.cast(
                tf.squeeze(batch['is_safe'] * batch['is_effective'], -1),
                tf.float32)
            valid = tf.identity(valid, 'valid')

            return {
                'state': state,
                'next_state': next_state,
                'action': action,
                'valid': valid,
            }

    def _state_loss(self,
                    targets,
                    outputs,
                    weights=1.0,
                    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS,
                    clip_value=None):
        pos_loss = 0.0
        for i in range(self.num_objects):
            pos_loss += loss_utils.l2_loss(
                targets=targets['position'][:, i, :] * POSITION_SCALE,
                outputs=outputs['position'][:, i, :] * POSITION_SCALE,
                weights=weights,
                reduction=reduction,
                clip_value=clip_value)
        pos_loss /= float(self.num_objects)

        return pos_loss * POSITION_SCALE

    def _action_loss(self,
                     targets,
                     outputs,
                     weights=1.0,
                     reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
        loss_start = loss_utils.l2_loss(
            targets=targets['start'],
            outputs=outputs['start'],
            weights=weights,
            reduction=reduction)
        loss_motion = loss_utils.l2_loss(
            targets=targets['motion'],
            outputs=outputs['motion'],
            weights=weights,
            reduction=reduction)
        return loss_start + loss_motion

    def _encoding_loss(self,
                       targets,
                       outputs,
                       weights=1.0,
                       reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
        cloud_loss = 0.0
        for i in range(self.num_objects):
            cloud_loss += loss_utils.l2_loss(
                targets=targets['encoded_cloud'][:, i, :],
                outputs=outputs['encoded_cloud'][:, i, :],
                weights=weights,
                reduction=reduction)
        cloud_loss /= float(self.num_objects)
        return cloud_loss

    def _add_state_summary(self, state, scope):
        with tf.variable_scope(scope):
            position = state['position']
            for i in range(self.num_objects):
                tf.summary.histogram('body%02d_x' % i, position[:, i, 0])
                tf.summary.histogram('body%02d_y' % i, position[:, i, 1])

    def _add_state_delta_summary(self, target_state, output_state, scope):
        with tf.variable_scope(scope):
            position = target_state['position'] - output_state['position']

            for i in range(self.num_objects):
                tf.summary.histogram('body%02d_dx' % i, position[:, i, 0])
                tf.summary.histogram('body%02d_dy' % i, position[:, i, 1])


class PushMPCProblem(PushProblem, mpc_problem.MPCProblem):

    def add_state_summaries(self, targets, outputs):
        """Add summaries of the reconstruction."""
        self._add_state_summary(outputs['pred_state'], 'next_state_output')
        self._add_state_delta_summary(
            targets['state'],
            outputs['pred_state'],
            'state_delta_output')

        return []


class PushSMPCProblem(PushProblem, mpc_problem.SMPCProblem):

    def add_state_summaries(self, targets, outputs):
        """Add summaries of the reconstruction."""
        self._add_state_summary(targets['next_state'], 'next_state_target')
        self._add_state_delta_summary(
            targets['state'],
            targets['next_state'],
            'state_delta_target')

        self._add_state_summary(outputs['pred_state'], 'next_state_output')
        self._add_state_delta_summary(
            targets['state'],
            outputs['pred_state'],
            'state_delta_output')

        self._add_state_summary(outputs['pred_state_high'], 'next_state_high')
        self._add_state_delta_summary(
            targets['state'],
            outputs['pred_state_high'],
            'state_delta_high')

        self._add_state_summary(outputs['pred_state_low'], 'next_state_low')
        self._add_state_delta_summary(
            targets['state'],
            outputs['pred_state_low'],
            'state_delta_low')

        return []

    def add_action_summaries(self, targets, outputs):
        with tf.variable_scope('action_target'):
            tf.summary.histogram('start', targets['action']['start'])
            tf.summary.histogram('motion', targets['action']['motion'])

        with tf.variable_scope('action_output'):
            tf.summary.histogram('start', outputs['action']['start'])
            tf.summary.histogram('motion', outputs['action']['motion'])

        return []


class PushVMPCProblem(PushProblem, baseline_problem.VMPCProblem):

    def add_state_summaries(self, targets, outputs):
        """Add summaries of the reconstruction."""
        self._add_state_summary(outputs['pred_state'], 'next_state_output')
        self._add_state_delta_summary(
            targets['state'],
            outputs['pred_state'],
            'state_delta_output')
        return []

    def add_action_summaries(self, targets, outputs):
        with tf.variable_scope('action_target'):
            tf.summary.histogram('start', targets['action']['start'])
            tf.summary.histogram('motion', targets['action']['motion'])

        with tf.variable_scope('action_output'):
            tf.summary.histogram('start', outputs['action']['start'])
            tf.summary.histogram('motion', outputs['action']['motion'])

        return []


class PushSectarProblem(PushProblem, baseline_problem.SectarProblem):

    def add_state_summaries(self, targets, outputs):
        """Add summaries of the reconstruction."""
        self._add_state_summary(outputs['pred_state'], 'next_state_output')
        self._add_state_delta_summary(
            targets['state'],
            outputs['pred_state'],
            'state_delta_output')
        return []

    def add_action_summaries(self, targets, outputs):
        with tf.variable_scope('action_target'):
            tf.summary.histogram('start', targets['action']['start'])
            tf.summary.histogram('motion', targets['action']['motion'])

        with tf.variable_scope('action_output'):
            tf.summary.histogram('start', outputs['action']['start'])
            tf.summary.histogram('motion', outputs['action']['motion'])

        return []
