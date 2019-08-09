import tensorflow as tf

def cost(x, params):
    start_x = tf.concat(
        [params['start'], x], axis=0)
    x_end = tf.concat(
        [x, params['end']], axis=0)
    
    f = tf.reduce_mean(
        tf.square(start_x - x_end))
    
    collision = tf.expand_dims(
        tf.transpose(
            params['collision'], [1, 0]), 0)
    x_expand = tf.expand_dims(x, 2)
    
    h = -tf.reduce_min(
        tf.linalg.norm(
            x_expand - collision, axis=1))
    return f + h

class Minimize(object):
    
    def __init__(self, 
                 func,
                 steps=1000, 
                 learning_rate=1e-2):
        self._func = func
        self._steps = steps
        self._lr = learning_rate
        return
    
    def __call__(self, x, params):
        _, x, params = tf.while_loop(
            cond=self._cond, 
            body=self._body, 
            loop_vars=[0, x, params])
        return x, params
    
    def _cond(self, step, x, params):
        c = tf.less(step, self._steps)
        return c
    
    def _body(self, step, x, params):
        cost = self._func(x, params)
        [grad] = tf.gradients(cost, x)
        x = x - grad * self._lr
        step = tf.identity(step)
        return tf.add(step, 1), x, params
