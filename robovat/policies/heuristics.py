"""Heuristics utilities for manipulation environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from robovat.utils.logging import logger


class HeuristicSampler(object):
    
    def __init__(self,
                 use_primitive,
                 cspace_low,
                 cspace_high,
                 motion_size,
                 translation_x,
                 translation_y,
                 rotation,
                 start_margin=0.04,
                 motion_margin=0.01,
                 max_attemps=100,
                 debug=None):
        self.use_primitive = use_primitive

        self.cspace_low = np.array(cspace_low)
        self.cspace_high = np.array(cspace_high)
        self.cspace_offset = 0.5 * (self.cspace_high + self.cspace_low)
        self.cspace_range = 0.5 * (self.cspace_high - self.cspace_low)

        self.motion_size = motion_size
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.rotation = rotation

        self.start_margin = start_margin
        self.motion_margin = motion_margin
        self.max_attemps = max_attemps

        self.debug = debug

    def sample(self, position, num_samples=1):
        list_start = []
        list_motion = []

        for i in range(num_samples):
            start, motion = self._sample(position)
            list_start.append(start)
            list_motion.append(motion)

        start = np.stack(list_start, axis=0)
        motion = np.stack(list_motion, axis=0)
        return start, motion

    def _sample(self, position):
        for i in range(self.max_attemps):
            start = np.random.uniform(-1., 1., [2])

            if self.use_primitive:
                motion = self.sample_motion()
            else:
                motion = np.random.uniform(-1., 1., [self.motion_size])

            waypoints = self.get_waypoints(start, motion)

            # Check if is_safe.
            if not self.is_waypoint_clear(
                    waypoints[0], None, position, self.start_margin):
                continue

            # Check if is_effective.
            is_effective = False
            # for i in range(1, 2):
            for i in range(1, len(waypoints)):
                if not self.is_waypoint_clear(
                        waypoints[i - 1], waypoints[i],
                        position, self.motion_margin):
                    is_effective = True
                    break

            if is_effective:
                break

        if i == self.max_attemps - 1:
            logger.info('HeuristicSampler did not find a good sample.')

        start = np.array(start, dtype=np.float32)
        motion = np.array(motion, dtype=np.float32)
        return start, motion

    def sample_motion(self):
        angle = np.random.uniform(0.0, 2 * np.pi)
        motion = 5 * [1.0 * np.cos(angle), 1.0 * np.sin(angle)]
        motion = np.array(motion, dtype=np.float32)
        motion += np.random.uniform(-0.3, 0.3, [self.motion_size])
        motion = np.clip(motion, -1.0, 1.0)
        return motion

    def get_waypoints(self, start, motion):
        motion = np.reshape(motion, [-1, 2])
        
        x = start[0] * self.cspace_range[0] + self.cspace_offset[0]
        y = start[1] * self.cspace_range[1] + self.cspace_offset[1]
        start = [x, y]
        waypoints = [start]

        for i in range(motion.shape[0]):
            delta_x = motion[i, 0] * self.translation_x
            delta_y = motion[i, 1] * self.translation_y

            x = x + delta_x
            y = y + delta_y

            x = np.clip(x, self.cspace_low[0], self.cspace_high[0])
            y = np.clip(y, self.cspace_low[1], self.cspace_high[1])

            waypoint = [x, y]
            waypoints.append(waypoint)

        return waypoints

    def is_waypoint_clear(self,
                          waypoint1,
                          waypoint2,
                          position,
                          margin,
                          debug=False):
        delta_x = position[..., 0] - waypoint1[0]
        delta_y = position[..., 1] - waypoint1[1]

        if waypoint2 is None:
            dist = np.sqrt(delta_x * delta_x + delta_y * delta_y)
            is_clear = dist > margin
        else:
            angle = np.arctan2(waypoint2[1] - waypoint1[1],
                               waypoint2[0] - waypoint1[0])
            waypoint_dist = np.linalg.norm([waypoint2[0] - waypoint1[0],
                                            waypoint2[1] - waypoint1[1]])
            boundary = [-margin, waypoint_dist + margin, -margin, margin]

            x = delta_x * np.cos(angle) + delta_y * np.sin(angle)
            y = delta_x * (-np.sin(angle)) + delta_y * np.cos(angle)

            is_x_clear = np.logical_or(x < boundary[0], x > boundary[1])
            is_y_clear = np.logical_or(y < boundary[2], y > boundary[3])
            is_clear = np.logical_or(is_x_clear, is_y_clear)

        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            
            plt.scatter(waypoint1[0], waypoint1[1], c='b')

            if waypoint2 is not None:
                plt.scatter(waypoint2[0], waypoint2[1], c='b')
                plt.plot([waypoint1[0], waypoint2[0]],
                         [waypoint1[1], waypoint2[1]],
                         c='b')

            for i in range(position.shape[0]):
                point = position[i]
                if is_clear[i]:
                    color = 'g'
                else:
                    color = 'r'
                plt.scatter(point[0], point[1], c=color)
                plt.text(point[0], point[1], '%d' % i)

            plt.subplot(1, 2, 2)
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            for i in range(position.shape[0]):
                point = position[i]
                if is_clear[i]:
                    color = 'g'
                else:
                    color = 'r'
                plt.scatter(x[i], y[i], c=color)
                plt.text(x[i], y[i], '%d' % i)

            plt.show()

        is_all_clear = np.all(is_clear)
        return is_all_clear
