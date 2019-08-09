"""YAML Configuration Parser.

Adapted from Jeff Mahler's code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from collections import OrderedDict

import yaml
from easydict import EasyDict as edict


class YamlConfig(object):
    """Class to load a configuration file and parse it into a dictionary.
    """

    def __init__(self, filename):
        """Initialize a YamlConfig by loading it from the given file.

        Args:
            filename: The filename of the .yaml file that contains the
                configuration.
        """
        self.config = None
        self._load_config(filename)

    def keys(self):
        """Return the keys of the config dictionary.

        Returns:
            A list of the keys in the config dictionary.
        """
        return self.config.keys()

    def update(self, d):
        """Update the config with a dictionary of parameters.

        Args:
            d: Dictionary of parameters.
        """
        self.config.update(d)

    def __contains__(self, key):
        """Overrides 'in' operator.
        """
        return key in self.config.keys()

    def __getitem__(self, key):
        """Overrides the key access operator [].
        """
        return self.config[key]

    def __setitem__(self, key, val):
        """Overrides the keyed setting operator [].
        """
        self.config[key] = val

    def iteritems(self):
        """Returns iterator over config dict.
        """
        return self.config.iteritems()

    def save(self, filename):
        """Save a YamlConfig to disk.
        """
        yaml.dump(self, open(filename, 'w'))

    def _load_config(self, filename):
        """Loads a yaml configuration file from the given filename.

        Args:
            filename: The filename of the .yaml file that contains the
                configuration.
        """
        # Read entire file for metadata.
        fh = open(filename, 'r')
        self.file_contents = fh.read()

        # Replace !include directives with content.
        config_dir = os.path.split(filename)[0]
        include_re = re.compile('^!include\s+(.*)$', re.MULTILINE)

        def include_repl(matchobj):
            fname = os.path.join(config_dir, matchobj.group(1))
            with open(fname) as f:
                return f.read()

        # TODO(kuanfang): Recursively replace the content.
        while re.search(include_re, self.file_contents):
            self.file_contents = re.sub(
                    include_re, include_repl, self.file_contents)

        # Read in dictionary.
        self.config = self.__ordered_load(self.file_contents)

        # Convert functions of other params to true expressions.
        for k in self.config.keys():
            self.config[k] = YamlConfig.__convert_key(self.config[k])

        fh.close()

        # Load core configuration.
        return self.config

    @staticmethod
    def __convert_key(expression):
        """Converts keys in YAML that reference other keys.
        """
        if (type(expression) is str and len(expression) > 2 and
                expression[1] == '!'):
            expression = eval(expression[2:-1])

        return expression

    def __ordered_load(self, stream, Loader=yaml.Loader,
                       object_pairs_hook=OrderedDict):
        """Load an ordered dictionary from a yaml file.
        """
        class OrderedLoader(Loader):
            pass

        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            lambda loader, node: object_pairs_hook(
                loader.construct_pairs(node)))

        return yaml.load(stream, OrderedLoader)

    def as_easydict(self):
        """Convert the configuration into the easydict format."""
        return edict(self.config)
