import numpy as np
import cvxpy as cvx


def solver_quadratic(target, force, s, d):
    Q = np.array([[1, 0, -1, 0],
                  [0, 1, 0, -1],
                  [-1, 0, 2, 0],
                  [0, -1, 0, 2]])
    xt, yt = target
    alpha, beta = force
    b = np.array([[-s * beta],
                  [s * alpha],
                  [-2 * xt + s * beta],
                  [-2 * yt - s * alpha]])
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


def solver_hammering(target, force, s, d, u):
    g_xy, g_rz = solver_quadratic(target, force, s, d)
    g_drz = -s * u / d
    g_drz = np.reshape(g_drz.astype(np.float32), ())
    return g_xy, g_rz, g_drz

