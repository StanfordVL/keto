"""File IO utilities using pickle.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from robovat.utils.logging import logger

try:
    import cPickle as pickle
except Exception as e:
    logger.warn(str(e))
    import pickle


class PickleWriter(object):
    """A class to dump pickle to file.
    """

    def __init__(self, filename):
        self._file = open(filename, 'wb')

    def write(self, data):
        """Write data to a pickle file.

        Args:
            data: An element of the data.
        """
        pickle.dump(data, self._file, protocol=pickle.HIGHEST_PROTOCOL)

    def close(self):
        self._file.close()


def read(filename):
    """Read data from a pickle file.

    Args:
        filename: The filename to the pickle file.

    Yields:
        data: An element of the data.
    """
    with open(filename, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                yield data
            except EOFError:
                break
