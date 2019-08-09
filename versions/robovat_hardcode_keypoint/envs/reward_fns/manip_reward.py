"""Reward function of the environments.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from easydict import EasyDict as edict

from robovat.envs.reward_fns import reward_fn
from robovat.utils.logging import logger


USE_SPARSE_REWARD = False


INSERT_GOAL_TILES = {
    'PATH': 'data/sim/envs/multi_stage/tile.urdf',
    'SIZE': 0.15,
    'SCOPE': 0.15,
    'RGBA': [0.867, 0.776, 0.678, 0.5],
    'SPECULAR': [0, 0, 0],
    'POSE': [0.6, -0.3, -0.05],
    'OFFSETS':
        [
         [0, 0, 0],
         ]
}
INSERT_GOAL_TILES = edict(INSERT_GOAL_TILES)


SLOT_TILES = {
    'PATH': 'data/sim/envs/multi_stage/tile.urdf',
    'SIZE': 0.15,
    'SCOPE': 0.15,
    'RGBA': [1, .4235, .4235, 0.1],
    'SPECULAR': [0, 0, 0],
    'POSE': [0.6, -0.3, 0],
    'OFFSETS':
        [
         [-2, 0, 0],
         [-1, 0, 0],
         [1, 0, 0],
         [2, 0, 0],

         [-2, -1, 0],
         [-1, -1, 0],
         [1, -1, 0],
         [2, -1, 0],

         [0, -1, 0],
         ]
}
SLOT_TILES = edict(SLOT_TILES)


CROSS_GOAL_TILES = {
    'PATH': 'data/sim/envs/multi_stage/tile.urdf',
    'SIZE': 0.15,
    'SCOPE': 0.15,
    # 'RGBA': [0.4235, 1, 0.4235, 0.1],
    'RGBA': [1, 0.9412, 0.4235, 0.1],
    'SPECULAR': [0, 0, 0],
    'POSE': [0.5, -0.3 + 0.075, 0.001],
    'OFFSETS':
        [
         [0, 0, 0],
         ]
}
CROSS_GOAL_TILES = edict(CROSS_GOAL_TILES)


PATH_TILES = {
    'PATH': 'data/sim/envs/multi_stage/tile.urdf',
    'SIZE': 0.15,
    'SCOPE': 0.15,
    # 'RGBA': [0.4667, 0.7098, 0.9961, 0.1],
    'RGBA': [0.8, 0.8, 0.8, 0.1],
    'SPECULAR': [0, 0, 0],
    'POSE': [0.5, -0.3 + 0.075, 0.001],
    'OFFSETS':
        [
         [-1, 0, 0],
         [0, -1, 0],
         [0, -2, 0],
         [-1, -2, 0],

         [1, 0, 0],
         [1, 1, 0],
         [1, 2, 0],
         [1, 3, 0],
         [0, 3, 0],
         [-1, 3, 0],
         [2, 0, 0],
         ]
}
PATH_TILES = edict(PATH_TILES)


WIDE_PATH_TILES = {
    'PATH': 'data/sim/envs/multi_stage/tile.urdf',
    'SIZE': 0.15,
    'SCOPE': 0.25,
    'RGBA': [0.4667, 0.7098, 0.9961, 0.1],
    'SPECULAR': [0, 0, 0],
    'POSE': [0.5, -0.3 + 0.075, 0.001],
    'OFFSETS':
        [
         [-1, 0, 0],
         [0, -1, 0],
         [0, -2, 0],
         [-1, -2, 0],

         [1, 0, 0],
         [1, 1, 0],
         [1, 2, 0],
         [1, 3, 0],
         [0, 3, 0],
         [-1, 3, 0],
         [2, 0, 0],
         ]
}
WIDE_PATH_TILES = edict(WIDE_PATH_TILES)


GATHER_TILES = {
    'PATH': 'data/sim/envs/multi_stage/tile.urdf',
    'SIZE': 0.15,
    'SCOPE': 0.15,
    'RGBA': [0.4667, 0.7098, 0.9961, 0.1],
    'SPECULAR': [0, 0, 0],
    'POSE': [0.6, 0, 0],
    'OFFSETS':
        [
         [-2, 0, 0],
         [-2, -1, 0],
         [-2, 1, 0],

         [-1, 0, 0],
         [-1, -1, 0],
         [-1, 1, 0],

         [0, 0, 0],
         [0, -1, 0],
         [0, 1, 0],

         # [1, 0, 0],
         # [1, -1, 0],
         # [1, 1, 0],

         ]
}
GATHER_TILES = edict(GATHER_TILES)
GATHER_THRESH = 0.15


def process_state(state):
    if isinstance(state, dict):
        state = state['position'][..., :2]
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)
    assert state.shape[-1] == 2
    return state


def get_reward_fn(task_name=None):
    if task_name is None:
        return dummy_reward_fn

    binary_fns = []

    if task_name == 'gather':
        unary_fn = get_gather_score
        # binary_fns = [get_stride_penalty]
    elif task_name == 'insert':
        unary_fn = get_insert_score
        binary_fns = [get_collision_penalty, get_stride_penalty]
    elif task_name == 'cross':
        unary_fn = get_cross_score
        binary_fns = [get_collision_penalty, get_stride_penalty]
    elif task_name == 'wide_cross':
        unary_fn = get_wide_cross_score
        binary_fns = [get_collision_penalty, get_stride_penalty]
    else:
        raise ValueError('Unrecognized manipulation task: %r' % task_name)

    def reward_fn(state, next_state):
        state = process_state(state)
        next_state = process_state(next_state)

        batch_size = state.shape[0]
        reward = np.zeros([batch_size], dtype=np.float32)
        termination = np.zeros([batch_size], dtype=np.bool)

        for binary_fn in binary_fns:
            binary_reward, binary_termination = binary_fn(state, next_state)
            reward += binary_reward
            termination = np.logical_or(termination, binary_termination)

        score, _ = unary_fn(state)
        next_score, unary_termination = unary_fn(next_state)
        unary_reward = next_score - score
        reward += unary_reward * np.logical_not(termination).astype(np.float32)
        termination = np.logical_or(termination, unary_termination)

        return reward, termination
    
    return reward_fn


def dummy_reward_fn(state, next_state):
    state = process_state(state)
    next_state = process_state(next_state)

    if len(state.shape) == 2:
        shape = ()
    elif len(state.shape) == 3:
        shape = (state.shape[0],)
    else:
        raise ValueError

    reward = np.ones(shape, dtype=np.float32)
    termination = np.zeros_like(reward, dtype=np.bool)
    return reward, termination


def get_kit_score(state, slot, peg_id):
    peg = state[:, peg_id]
    score = -np.linalg.norm(slot - peg, axis=-1)
    termination = np.zeros_like(score, dtype=np.bool)
    return score, termination


def get_tile_positions(tile_config):
    center = tile_config.POSE
    size = tile_config.SIZE
    tile_positions = []
    for i, offset in enumerate(tile_config.OFFSETS):
        position = np.array(center) + np.array(offset) * size
        tile_positions.append(position[..., :2])
    tile_positions = np.stack(tile_positions, axis=0)
    return tile_positions


def is_on_tiles(position, tile_config):
    tiles = get_tile_positions(tile_config)
    scope = tile_config.SCOPE
    assert len(position.shape) == 2
    assert len(tiles.shape) == 2
    dists = np.expand_dims(position, axis=1) - np.expand_dims(tiles, axis=0)
    is_on_tiles = np.logical_and(np.abs(dists[:, :, 0]) <= 0.5 * scope,
                                 np.abs(dists[:, :, 1]) <= 0.5 * scope)
    return np.any(is_on_tiles, axis=1)


def get_tile_dists(position, tile_config):
    tiles = get_tile_positions(tile_config)
    dists = np.expand_dims(position, axis=1) - np.expand_dims(tiles, axis=0)
    dists = np.linalg.norm(dists, axis=-1)
    dists = dists.min(axis=1)
    return dists


def get_collision_penalty(state, next_state, min_dist=0.1,
                          weight=10.0):
    target = next_state[:, 0, :]
    dists = np.expand_dims(target, axis=1) - next_state[:, 1:, :]
    dists = np.linalg.norm(dists, axis=-1)
    collision = np.less(dists, min_dist)
    reward = -weight * np.sum(collision.astype(np.float32), axis=1)
    termination = np.any(collision, axis=1)
    termination = np.zeros_like(termination)  # TODO(debug)
    return reward, termination


def get_stride_penalty(state, next_state, min_dist=0.1, max_dist=0.3,
                       weight=10.0):
    strides = np.linalg.norm(next_state - state, axis=-1)
    violate = np.logical_or(
        np.all(np.less(strides, min_dist), axis=1),
        np.any(np.greater(strides, max_dist), axis=1)
    )

    reward = -weight * violate.astype(np.float32)
    termination = violate
    termination = np.zeros_like(termination)  # TODO(debug)
    return reward, termination


def get_gather_score(state):
    num_bodies = state.shape[1]
    for i in range(num_bodies):
        position = state[:, i, :]
        if i == 0:
            invalid = is_on_tiles(position, GATHER_TILES)
        else:
            invalid = np.logical_or(
                invalid,
                is_on_tiles(position, GATHER_TILES))

    if USE_SPARSE_REWARD: 
        goal_dists = 0.0
    else:
        goal = np.stack(
            [0.7 * np.ones_like(state[..., 0]),
             np.zeros_like(state[..., 1])],
            axis=-1
        )
        dists = np.linalg.norm(state[..., 0:1] - goal[..., 0:1], axis=-1)
        goal_dists = np.mean(dists, axis=-1)

    goal_reached = np.logical_and(
        np.less(goal_dists, GATHER_THRESH),
        np.logical_not(invalid)
    )
    
    score = (
        0.0
        - goal_dists * 10.
        + 1000. * goal_reached.astype(np.float32)
    )
    termination = goal_reached.astype(np.bool)
    return score, termination


def get_insert_score(state):
    target = state[:, 0, :]
    goal_reached = is_on_tiles(target, INSERT_GOAL_TILES)

    num_bodies = state.shape[1]
    for i in range(num_bodies):
        position = state[:, i, :]
        if i == 0:
            invalid = is_on_tiles(position, SLOT_TILES)
        else:
            invalid = np.logical_or(
                invalid,
                is_on_tiles(position, SLOT_TILES))

    if USE_SPARSE_REWARD: 
        goal_dists = 0.0
    else:
        goal_dists = get_tile_dists(target, INSERT_GOAL_TILES)

    score = (
        0.0
        - goal_dists * 10.
        - 100. * invalid.astype(np.float32)
        + 1000. * goal_reached.astype(np.float32)
    )
    termination = np.logical_or(
        invalid.astype(np.bool),
        goal_reached.astype(np.bool))

    return score, termination


def get_cross_score(state, path_tiles=PATH_TILES):
    target = state[:, 0, :]
    goal_reached = is_on_tiles(target, CROSS_GOAL_TILES)

    invalid = np.logical_not(is_on_tiles(target, path_tiles))

    if USE_SPARSE_REWARD: 
        goal_dists = 0.0
    else:
        goal_dists = get_tile_dists(target, CROSS_GOAL_TILES)

    score = (
        0.0
        - goal_dists * 10.
        - 100. * invalid.astype(np.float32)
        + 1000. * goal_reached.astype(np.float32)
    )
    termination = np.logical_or(
        invalid.astype(np.bool),
        goal_reached.astype(np.bool))
    return score, termination


def get_wide_cross_score(state):
    return get_cross_score(state, path_tiles=WIDE_PATH_TILES)


class ManipReward(reward_fn.RewardFn):
    """Reward function of the environments."""
    
    def __init__(self,
                 name,
                 task_name=None,
                 streaming_length=1000):
        """Initialize."""
        self.name = name 
        self.task_name = task_name
        self.streaming_length = streaming_length

        self.env = None
        self.reward_fn = get_reward_fn(task_name)

        self.history = []

    def get_reward(self):
        """Returns the reward value of the current step."""
        if self.env.simulator:
            assert self.env.prev_obs_data is not None
            assert self.env.obs_data is not None
            reward, termination = self.reward_fn(self.env.prev_obs_data,
                                                 self.env.obs_data)
        else:
            raise NotImplementedError

        self._update_history(reward)

        # if self.env.debug:
        #     avg_reward = np.mean(self.history or [-1])
        #     logger.debug('reward: %.3f, avg_reward %.3f', reward, avg_reward)

        if self.env.simulator.check_contact(
                [self.env.ground, self.env.robot.base],
                self.env.movable_bodies):
            logger.warning('Unsafe action: Movable bodies drop to ground.')
            termination = np.ones_like(termination, dtype=np.bool)

        reward = float(reward)
        termination = bool(termination)

        return reward, termination

    def _update_history(self, reward):
        self.history.append(reward)

        if len(self.history) > self.streaming_length:
            self.history = self.history[-self.streaming_length:]
