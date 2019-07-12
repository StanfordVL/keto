# coding= utf8

import math
import warnings

import scipy.optimize
import numpy as np


EPS = 1e-14


def quaternion_from_matrix3(matrix3):
    """Return quaternion from 3x3 rotation matrix.
    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix3, dtype=np.float64, copy=False)[:3, :3]
    t = np.trace(M) + 1
    if t <= -EPS:
        warnings.warn('Numerical warning of [t = np.trace(M) + 1 = %f]'
                      % (t))
    t = max(t, EPS)
    q[3] = t
    q[2] = M[1, 0] - M[0, 1]
    q[1] = M[0, 2] - M[2, 0]
    q[0] = M[2, 1] - M[1, 2]
    q *= 0.5 / math.sqrt(t)
    return q


def inverse_kinematic_optimization(chain,
                                   target,
                                   initial_positions,
                                   max_iter=None):
    """Computes the inverse kinematic on the specified target.

    :param ikpy.chain.Chain chain: The chain used for the Inverse kinematics.
    :param numpy.array target: The desired target.
    :param numpy.array initial_positions: The initial pose of your chain.
    :param int max_iter: Maximum number of iterations for the optimisation
        algorithm.
    """
    if initial_positions is None:
        raise ValueError("initial_positions must be specified")

    # Compute squared distance to target
    def objective(x):
        y = chain.active_to_full(x, initial_positions)
        forward = chain.forward_kinematics(y)
        objective_position = np.linalg.norm(
                target[:3, 3] - forward[:3, 3])
        target_quaternion = quaternion_from_matrix3(target[:3, :3])
        forward_quaternion = quaternion_from_matrix3(forward[:3, :3])
        objective_orientation = -np.dot(
                target_quaternion,
                forward_quaternion)
        return objective_position + objective_orientation

    # Compute bounds
    bounds = [link.bounds for link in chain.links]
    bounds = chain.active_from_full(bounds)

    options = {}
    # Manage iterations maximum
    if max_iter is not None:
        options["maxiter"] = max_iter

    # Utilisation d'une optimisation L-BFGS-B
    result = scipy.optimize.minimize(
            objective,
            chain.active_from_full(initial_positions),
            method='L-BFGS-B',
            bounds=bounds,
            options=options)

    return chain.active_to_full(result.x, initial_positions)
