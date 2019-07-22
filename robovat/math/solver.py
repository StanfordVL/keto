import tensorflow as tf

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
