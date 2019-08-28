import numpy as np
import cvxpy as cvx


def solver_quadratic(target, force, theta, d):
    Q = np.array([[1, 0, -1, 0],
                  [0, 1, 0, -1],
                  [-1, 0, 2, 0],
                  [0, -1, 0, 2]])
    xt, yt = target
    alpha, beta = force
    b = np.array([[0 - alpha*np.cos(theta) - beta*np.sin(theta)],
                  [0 + alpha*np.sin(theta) - beta*np.cos(theta)],
                  [-2 * xt + alpha*np.cos(theta) + beta*np.sin(theta)],
                  [-2 * yt - alpha*np.sin(theta) + beta*np.cos(theta)]])
    c = np.array([[0], [0], [0], [0]])
    x = cvx.Variable([4, 1])
    obj = cvx.Minimize(cvx.quad_form(x, Q) + b.T * x)
    constraints = [x >= c]
    p = cvx.Problem(obj, constraints)
    p.solve()
    x = x.value.flatten()
    vect = x[:2] - x[2:]
    vect = vect / (1e-6 + np.sqrt(np.sum(vect * vect)))
    g_xy = x[2:] + vect * d
    g_rz = np.arctan2(-vect[1], -vect[0])
    g_xy = g_xy.astype(np.float32)
    g_rz = g_rz.astype(np.float32)
    return g_xy, g_rz


def solver_hammering(target, force, theta, d, u):
    g_xy, g_rz = solver_quadratic(target, force, theta, d)
    g_drz = u / (1e-4 + d)
    g_drz = np.reshape(g_drz.astype(np.float32), ())
    return g_xy, g_rz, g_drz


def solver_general(target, force, theta, d):
    return solver_quadratic(target, force, theta, d)
