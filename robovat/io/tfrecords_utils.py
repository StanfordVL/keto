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

        if not os.path.isdir(output_dir):
            logger.info('Making output directory %s...', output_dir)
            os.makedirs(output_dir)

    def __call__(self, trajectory):
        # Create a file for saving the episode data.
        if self._num_entries_this_file == 0:
            if self._use_random_name:
                timestamp = time_utils.get_timestamp_as_string()
            else:
                timestamp = '%06d' % (self._num_files)
            filename = 'episodes_%s.tfrecords' % (timestamp)
            self._output_path = os.path.join(self._output_dir, filename)
            self._num_files += 1
            if self._writer:
                self._writer.close()
            self._writer = tf.python_io.TFRecordWriter(self._output_path)

        # Append the episode to the file.
        logger.info('Saving trajectory to file %s (%d / %d)...',
                    self._output_path,
                    self._num_entries_this_file,
                    self._num_entries_per_file)
        num_entries = self.write(trajectory)

        # Update the cursor.
        self._num_entries_this_file += num_entries
        self._num_entries_this_file %= self._num_entries_per_file
        self._num_calls = self._num_calls + 1

    def write(self, trajectory):
        """Write a string record to the file."""
        data, extra_data = self._problem.convert_trajectory(trajectory)

        if not isinstance(data, list):
            data = [data]

        for entry in data:
            self._write_data(entry)

        if extra_data:
            self._write_extra_data(extra_data)

        return len(data)

    def _write_data(self, data):
        feature = dict()
        
        for name, spec in self._spec.items():
            element = data[name]
            if isinstance(spec, OrderedDict):
                for sub_key, sub_spec in spec.items():
                    sub_name = '%s/%s' % (name, sub_key)
                    write_feature(feature, sub_name, sub_spec, element[sub_key])
            else:
                write_feature(feature, name, spec, element)

        example = tf.train.Example(
                features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()
        self._writer.write(serialized)
        self._writer.flush()

    def _write_extra_data(self, data):
        for name in data.keys():
            item = data[name]
            out_dir = os.path.join(
                          self._output_dir,
                          name)
            if not os.path.exists(out_dir):
                os.system('mkdir -p ' + out_dir)
            path = os.path.join(out_dir, 
                    '%06d.npy' % (self._num_calls))
            with open(path, 'wb') as f:
                np.save(f, item)
        return

    def close(self):
        if self._writer is not None:
            self._writer.close()


class TrajectoryReader(object):
    """Read trajectories from TFRecord files."""

    def __init__(self,
                 problem,
                 filename,
                 num_epochs=None):
        self._problem = problem
        self._filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=num_epochs)
        self._reader = tf.TFRecordReader(filename)

        self._spec = problem.spec

        self._keys_to_features = dict()
        for name, spec in self._spec.items():
            if isinstance(spec, OrderedDict):
                for sub_key, sub_spec in spec.items():
                    sub_name = '%s/%s' % (name, sub_key)
                    parse_feature(self._keys_to_features, sub_name, sub_spec)
            else:
                parse_feature(self._keys_to_features, name, spec)

    def read(self):
        """Read a string record from the file.

        Returns:
            features: A dictionary of feature tensors.
        """
        _, serialized_example = self._reader.read(self._filename_queue)

        # Parse features.
        parsed_features = tf.parse_single_example(  
                serialized_example, features=self._keys_to_features)  
        features = dict()

        for name, spec in self._spec.items():
            if isinstance(spec, OrderedDict):
                for sub_key, sub_spec in spec.items():
                    sub_name = '%s/%s' % (name, sub_key)
                    decode_feature(
                        parsed_features, features, sub_name, sub_spec)
            else:
                decode_feature(parsed_features, features, name, spec)

        # Observation.
        observation = OrderedDict()
        for name, feature in features.items():
            if 'observation/' == name[:12]:
                observation[name[12:]] = feature
        if len(observation) == 0:
            observation = features['observation']
        
        # Action.
        action = OrderedDict()
        for name, feature in features.items():
            if 'action/' == name[:7]:
                action[name[7:]] = feature
        if len(action) == 0:
            action = features['action']

        return Trajectory(
            step_type=None,
            observation=observation,
            action=action,
            policy_info=None,
            next_step_type=None,
            reward=features['reward'],
            discount=features['discount'])
