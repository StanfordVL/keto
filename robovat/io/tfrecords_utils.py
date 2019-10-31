"""Trajectory writer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tf_agents.environments.trajectory import Trajectory

from robovat.utils import time_utils
from robovat.utils.logging import logger


def bytes_feature(value):
    value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    value = np.array(value)
    if value.shape == ():
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    value = np.array(value)
    if value.shape == ():
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_feature(feature, name, spec, data):
    shape = spec.shape
    dtype = spec.dtype
    value = np.array(data)

    if shape.ndims > 1:
        image_data = value.tostring()
        feature[name] = bytes_feature(image_data)
    elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool:
        feature[name] = int64_feature(value)
    elif dtype.is_floating:
        feature[name] = float_feature(value)
    else:
        raise ValueError('Unrecognized feature %s of shape %r and type %r.'
                         % (name, shape, dtype))

    return feature


def parse_feature(keys_to_features, name, spec):
    shape = spec.shape
    dtype = spec.dtype

    if shape.ndims > 1:
        keys_to_features[name] = tf.FixedLenFeature([], tf.string)
    elif dtype.is_integer or dtype.is_unsigned or dtype.is_bool:
        keys_to_features[name] = tf.FixedLenFeature(shape, tf.int64)
    elif dtype.is_floating:
        keys_to_features[name] = tf.FixedLenFeature(shape, tf.float32)
    else:
        raise ValueError('Unrecognized feature %s of shape %r and type %r.'
                         % (name, shape, dtype))


def decode_feature(parsed_features, features, name, spec):
    shape = spec.shape
    dtype = spec.dtype

    feature = parsed_features[name]
    if shape.ndims > 1:
        feature = tf.decode_raw(feature, tf.float32)
    feature = tf.reshape(feature, shape)
    feature = tf.cast(feature, dtype)

    features[name] = feature


class TrajectoryWriter(object):
    """Write trajecoties as TFRecord format."""

    def __init__(self,
                 problem,
                 output_dir,
                 num_entries_per_file=1000,
                 use_random_name=True):
        self._problem = problem
        self._output_dir = output_dir
        self._num_entries_per_file = num_entries_per_file
        self._use_random_name = use_random_name

        self._spec = problem.spec

        self._writer = None
        self._output_path = None
        self._num_files = 0
        self._num_entries_this_file = 0
        self._num_calls = 0

        self._process_id = np.random.randint(1e+6)

        if not os.path.isdir(output_dir):
            logger.info('Making output directory %s...', output_dir)
            os.makedirs(output_dir)

    def __call__(self, trajectory):
        self.write(trajectory)
        self._num_calls = self._num_calls + 1
        return

    def write(self, trajectory):
        """Write a record to the file."""
        _, extra_data = self._problem.convert_trajectory(trajectory)

        if extra_data:
            self._write_extra_data(extra_data)
        return

    def _write_extra_data(self, data):
        for name in data.keys():
            item = data[name]
            out_dir = os.path.join(
                          self._output_dir,
                          name)
            if not os.path.exists(out_dir):
                os.system('mkdir -p ' + out_dir)
            path = os.path.join(out_dir, 
                    '{}_{}.npy'.format(
                        str(self._process_id).zfill(6),
                        str(self._num_calls).zfill(6)))
            with open(path, 'wb') as f:
                np.save(f, item)
        return

    def close(self):
        if self._writer is not None:
            self._writer.close()

