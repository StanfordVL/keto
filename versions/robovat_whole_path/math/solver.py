import tensorflow as tf


def solve_actions_rot(start_xyz,
                      start_rz,
                      target_xyz,
                      target_dxyz,
                      func_r,
                      steps,
                      self_c_kp,
                      env_c_kp,
                      max_iter=1000,
                      learning_rate=1e-3,
                      x_init_offsets=[0.0, 0.0, 0.0]):
    """Solve intermediate poses
    """
    def _reshape(xs, num_feats):
        out = [tf.reshape(x, [-1, num_feat])
               for x, num_feat in zip(xs, num_feats)]
        return out

    def _rot_mat(self, rz):
        zero = tf.constant(0.0,
                           dtype=tf.float32)
        one = tf.constant(1.0,
                          dtype=tf.float32)
        mat = tf.reshape(
            [[tf.cos(rz), -tf.sin(rz), zero],
             [tf.sin(rz), tf.cos(rz), zero],
             [zero, zero, one]], [3, 3])
        return mat

    def _init(start, target, steps):
        """Linear initial poses
        """
        delta = (target - start) / (steps + 1)
        dx_init = delta * tf.cast(tf.reshape(
            tf.range(1, 1 + steps),
            [steps, 1]), tf.float32)
        return dx_init + start

    def _distance_l2(x, y):
        return tf.reduce_mean(tf.square(x - y))

    [start_xyz, target_xyz, target_dxyz,
     self_c_kp, env_c_kp] = _reshape(
        [start_xyz, target_xyz, target_dxyz,
         self_c_kp, env_c_kp],
        [3, 3, 3, 3, 3])

    xyz_init = _init(start_xyz, target_xyz, steps)
    rz_init = _init(start_rz, start_rz, steps)

    self_c_kp = tf.reshape(self_c_kp, [1, -1, 1, 3])
    env_c_kp = tf.reshape(env_c_kp, [-1, 1, 1, 3])

    def _cost(xyz, rz):
        zero = [tf.constant(0.0, tf.float32)]
        func_endxyz = tf.concat(
            [func_r * tf.cos(rz[-1]),
             func_r * tf.sin(rz[-1]), zero], axis=0)
        func_endxyz = func_endxyz + xyz[-1]
        g = tf.linalg.norm(func_endxyz - target_xyz)

        f_xyz = _distance_l2(
            tf.concat(
                [[start_xyz], tf.squeeze(xyz)], 
                axis=0)[:-1],
            tf.squeeze(xyz))

        f_rz_sin = _distance_l2(
                tf.sin(tf.concat(
                    [[start_rz], tf.squeeze(rz)], 
                    axis=0)[:-1]),
                tf.sin(tf.squeeze(rz)))

        f_rz_cos = _distance_l2(
                tf.cos(tf.concat(
                    [[start_rz], tf.squeeze(rz)], 
                    axis=0)[:-1]),
                tf.cos(tf.squeeze(rz)))

        func_xyz_endvect = tf.concat(
            [func_r * tf.cos(rz[-2]),
             func_r * tf.sin(rz[-2]), zero], axis=0)
        func_xyz_endvect = func_xyz_endvect + xyz[-2]
        func_xyz_endvect = func_endxyz - func_xyz_endvect
        f_endvect = -tf.reduce_sum(
                func_xyz_endvect * target_dxyz)

        f = f_xyz + f_rz_sin + f_rz_cos + f_endvect

        cost = f + g
        return cost

    def _cond(iter, xyz, rz):
        c = tf.less(iter, max_iter)
        return c

    def _body(iter, xyz, rz):
        cost = _cost(xyz, rz)

        [grad_xyz] = tf.gradients(cost, xyz)
        mask = tf.constant([1, 1, 0],
                           dtype=tf.float32)
        mask = tf.reshape(mask, [1, 3])
        grad_xyz = grad_xyz * mask
        xyz = xyz - grad_xyz * learning_rate

        grad_rz = tf.gradients(cost, rz)
        rz = rz - grad_rz * learning_rate

        iter = tf.identity(iter)
        return tf.add(iter, 1), xyz, rz

    _, xyz, rz = tf.while_loop(
        cond=_cond,
        body=_body,
        loop_vars=[0, xyz_init, rz_init])

    xyz_rz = tf.concat([xyz, rz], axis=1)
    return xyz_rz


def solve_actions(start, target, steps,
                  self_c_kp, env_c_kp,
                  max_iter=1000,
                  learning_rate=1e-3,
                  x_init_offsets=[0.0, 0.0, 0.0]):
    """Solve intermediate poses
    """
    def _reshape(xs, num_feats):
        out = [tf.reshape(x, [-1, num_feat])
               for x, num_feat in zip(xs, num_feats)]
        return out

    def _init(start, target, steps):
        """Linear initial poses
        """
        delta = (target - start) / (steps + 1)
        dx_init = delta * tf.cast(tf.reshape(
            tf.range(1, 1 + steps),
            [steps, 1]), tf.float32)
        return dx_init + start

    [start, target, self_c_kp, env_c_kp
     ] = _reshape([start, target,
                   self_c_kp, env_c_kp],
                  [4, 4, 3, 3])

    x_init = _init(start, target, steps)
    x_init, rot = tf.split(x_init, [3, 1], axis=1)
    x_init_offsets = tf.reshape(
        tf.constant(
            x_init_offsets,
            dtype=tf.float32), [1, 3])
    x_init = x_init + x_init_offsets

    x_init = tf.reshape(x_init, [1, 1, -1, 3])

    self_c_kp = tf.reshape(self_c_kp, [1, -1, 1, 3])
    env_c_kp = tf.reshape(env_c_kp, [-1, 1, 1, 3])

    def _cost(x):
        dist = tf.linalg.norm(
            x + self_c_kp - env_c_kp, axis=3)
        h = -tf.reduce_mean(dist)

        start_x = tf.concat(
            [start[:, :3],
             tf.squeeze(x)], axis=0)
        x_target = tf.concat(
            [tf.squeeze(x),
             target[:, :3]], axis=0)
        f = tf.reduce_max(
            tf.linalg.norm(start_x - x_target,
                           axis=1))
        cost = 2.0 * f + h
        return cost

    def _cond(iter, x):
        c = tf.less(iter, max_iter)
        return c

    def _body(iter, x):
        cost = _cost(x)
        [grad] = tf.gradients(cost, x)
        mask = tf.constant([1, 1, 0],
                           dtype=tf.float32)
        mask = tf.reshape(mask, [1, 1, 1, 3])
        grad = grad * mask
        x = x - grad * learning_rate
        iter = tf.identity(iter)
        return tf.add(iter, 1), x

    _, x = tf.while_loop(
        cond=_cond,
        body=_body,
        loop_vars=[0, x_init])

    x = tf.concat([tf.squeeze(x), rot], axis=1)
    return x
