"""File IO of TFRecord format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cStringIO as StringIO
import numpy as np
import tensorflow as tf
import PIL.Image


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


class TFRecordWriter(object):
    """A class to write TFRecords to file.
    """

    def __init__(self, filename, config):
        self._writer = tf.python_io.TFRecordWriter(filename)
        self._config = config

    def write(self, data):
        """Write a string record to the file.

        Args:
            data: A dictionary of numpy arrays.
        """
        feature = dict()

        for name, (dtype, shape) in self._config.items():
            value = np.array(data[name])

            if dtype == 'int64':
                feature[name] = int64_feature(value)
            elif dtype == 'float':
                feature[name] = float_feature(value)
            elif dtype == 'raw':
                image_data = value.tostring() 
                feature['%s/encoded' % (name)] = bytes_feature(image_data)
                feature['%s/format' % (name)] = bytes_feature(b'raw')
            elif dtype == 'png':
                pil_image = PIL.Image.fromarray(value)
                output = StringIO.StringIO()
                pil_image.save(output, format='PNG')
                image_data = output.getvalue()
                feature['%s/encoded' % (name)] = bytes_feature(image_data)
                feature['%s/format' % (name)] = bytes_feature('png')
            else:
                raise ValueError('Unrecognized feature type %s of key %s'
                                 % (dtype, name))

        example = tf.train.Example(
                features=tf.train.Features(feature=feature))
        serialized = example.SerializeToString()
        self._writer.write(serialized)

    def close(self):
        self._writer.close()


class TFRecordReader(object):
    """A class to read TFRecords from file.
    """

    def __init__(self, filename, config, num_epochs=None):
        self._filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=num_epochs)
        self._reader = tf.TFRecordReader(filename)
        self._config = config

    def read(self):
        """Read a string record from the file.

        Returns:
            features: A dictionary of feature tensors.
        """
        _, serialized_example = self._reader.read(self._filename_queue)

        # Define the feature formats.
        keys_to_features = dict()
        for name, (dtype, shape) in self._config.items():
            if dtype == 'int64':
                keys_to_features[name] = tf.FixedLenFeature(shape, tf.int64)
            elif dtype == 'float':
                keys_to_features[name] = tf.FixedLenFeature(shape, tf.float32)
            elif dtype == 'raw':
                keys_to_features['%s/encoded' % (name)] = tf.FixedLenFeature(
                        [], tf.string)
            elif dtype == 'png':
                keys_to_features['%s/encoded' % (name)] = tf.FixedLenFeature(
                        [], tf.string)
            else:
                raise ValueError('Unrecognized feature type %s of key %s'
                                 % (dtype, name))

        # Parse features.
        features = tf.parse_single_example(  
                serialized_example, features=keys_to_features)  

        # Decode, cast, reshape features.
        for name, (dtype, shape) in self._config.items():
            if dtype == 'int64':
                feature = features[name]
                feature = tf.cast(feature, tf.int32)
            elif dtype == 'float':
                feature = features[name]
                feature = tf.cast(feature, tf.float32)
            elif dtype == 'raw':
                feature = features['%s/encoded' % (name)]
                feature = tf.decode_raw(feature, tf.float32)
                feature = tf.reshape(feature, shape)
            elif dtype == 'png':
                feature = features['%s/encoded' % (name)]
                feature = tf.image.decode_image(feature, shape[-1])
                feature = tf.reshape(feature, shape)
            else:
                raise ValueError('Unrecognized feature type %s of key %s'
                                 % (dtype, name))

            features[name] = feature

        return features
