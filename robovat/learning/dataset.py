#!/usr/bin/env python

"""Provides data for the gym dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


# TODO(kuanfang): After we have more stuffs for learning, we should move this
# file to a learning folder.


class ImageHandler(slim.tfexample_decoder.ItemHandler):
    """An ItemHandler that decodes a parsed Tensor as an image."""

    def __init__(self,
                 image_key=None,
                 image_format=None,
                 shape=None,
                 channels=3,
                 dtype=tf.uint8,
                 repeated=False):
        """Initializes the image.
        Args:
            image_key: the name of the TF-Example feature in which the encoded
                image is stored.
            format_key: the name of the TF-Example feature in which the image
              format is stored.
            shape: the output shape of the image as 1-D `Tensor` [height, width,
                channels]. If provided, the image is reshaped accordingly. If
                left as None, no reshaping is done. A shape should be supplied
                only if all the stored images have the same shape.
            channels: the number of channels in the image.
            dtype: images will be decoded at this bit depth. Different formats
                support different bit depths.
            repeated: if False, decodes a single image. If True, decodes a
                variable number of image strings from a 1D tensor of strings.
        """
        if not image_key:
            image_key = 'image/encoded'

        super(ImageHandler, self).__init__([image_key])
        self._image_key = image_key
        self._image_format = image_format
        self._shape = shape
        self._channels = channels
        self._dtype = dtype
        self._repeated = repeated

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        image_buffer = keys_to_tensors[self._image_key]

        if self._repeated:
            return tf.map_fn(
                    lambda x: self._decode(x, self._image_format),
                    image_buffer, dtype=self._dtype)
        else:
            return self._decode(image_buffer, self._image_format)

    def _decode(self, image_buffer, image_format):
        """Decodes the image buffer.
        Args:
            image_buffer: The tensor representing the encoded image tensor.
            image_format: The image format for the image in `image_buffer`. If
                image format is `raw`, all images are expected to be in this
                format, otherwise this op can decode a mix of `jpg` and `png`
                formats.
        Returns:
            A tensor that represents decoded image of self._shape, or (?, ?,
                self._channels) if self._shape is not specified.
        """
        def decode_image():
            """Decodes a png or jpg based on the headers."""
            return tf.image.decode_image(image_buffer, self._channels)

        def decode_raw():
            """Decodes a raw image."""
            return tf.decode_raw(image_buffer, out_type=self._dtype)

        if self._image_format == 'raw' or self._image_format == 'RAW':
            image = decode_raw()
        else:
            image = decode_image()
            image.set_shape([None, None, self._channels])

        if self._shape is not None:
            image = tf.reshape(image, self._shape)

        return image


def get_dataset(config, path):
    """Gets a dataset tuple with instructions for reading flowers.

    Args:
        config: Configuration of the features.
        path: Path to the files.

    Returns:
        A `Dataset` namedtuple.
    """
    # Define keys_to_features.
    keys_to_features = dict()
    items_to_handlers = dict()

    for name, (dtype, shape) in config.items():
        if dtype == 'int64':
            keys_to_features[name] = tf.FixedLenFeature(shape, tf.int64)
            handler = slim.tfexample_decoder.Tensor(name, shape=shape)
        elif dtype == 'float':
            keys_to_features[name] = tf.FixedLenFeature(shape, tf.float32)
            handler = slim.tfexample_decoder.Tensor(name, shape=shape)
        elif dtype == 'raw':
            keys_to_features['%s/encoded' % (name)] = tf.FixedLenFeature(
                    [], tf.string)
            keys_to_features['%s/format' % (name)] = tf.FixedLenFeature(
                    [], tf.string)
            handler = ImageHandler(
                    '%s/encoded' % (name), 'raw',
                    shape=shape, channels=None, dtype=tf.float32)
        elif dtype == 'png':
            keys_to_features['%s/encoded' % (name)] = tf.FixedLenFeature(
                    [], tf.string)
            keys_to_features['%s/format' % (name)] = tf.FixedLenFeature(
                    [], tf.string)
            handler = slim.tfexample_decoder.Image(
                    '%s/encoded' % (name), '%s/format' % (name),
                    shape=shape, channels=shape[-1])
        else:
            raise ValueError('Unrecognized feature type %s of key %s'
                             % (dtype, name))

        items_to_handlers[name] = handler

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=path,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=None,
        items_to_descriptions=None)


def provide_batch(config, path, batch_size, num_threads=4, shuffle=True):
    """Provides a batch of images and corresponding labels.
    Args:
        config: Configuration of the features.
        path: Path to the files.
        batch_size: The batch size.
        shuffle: If shuffle the data.

    Returns:
        A batch dictionary mapping names to tensors.
    """
    dataset = get_dataset(config, path)

    provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=shuffle,
            num_readers=num_threads,
            common_queue_capacity=20 * batch_size,
            common_queue_min=10 * batch_size)

    data = dict(zip(config.keys(), provider.get(config.keys())))

    batch = tf.train.batch(
            data,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=5 * batch_size)

    return batch
