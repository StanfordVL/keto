"""Base class of sampler used in CEM algorithm."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class Sampler(abc.ABCNeta):
    """Base class of sampler used in CEM algorithm."""

    def __call__(observation
