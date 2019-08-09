"""Dummy class of Trajectory Problem.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

from robovat.problems import problem


class TrajectoryProblem(problem.Problem):
    """Trajectory Problem."""

    def __init__(self, time_step_spec, action_spec):
        """Initialize."""
        self.time_step_spec = time_step_spec
        self.action_spec = action_spec

    @property
    def spec(self):
        """Tensor spec of the problem as an OrderedDict."""
        return OrderedDict([
            ('observation', self.time_step_spec.observation),
            ('action', self.action_spec),
            ('reward', self.time_step_spec.reward),
            ('discount', self.time_step_spec.discount),
        ])

    def convert_trajectory(self, trajectory):
        """Convert trajectory."""
        return OrderedDict([
            ('observation', trajectory.observation),
            ('action', trajectory.action),
            ('reward', trajectory.reward),
            ('discount', trajectory.discount),
        ])
