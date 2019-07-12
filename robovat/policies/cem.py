"""Cross-Entropy Method (CEM).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

nest = tf.contrib.framework.nest
tfd = tfp.distributions


class CEM(object):
    """Cross-Entropy Method (CEM)."""

    def __init__(self,
                 func,
                 num_samples=64,
                 num_elites=6,
                 num_iterations=3,
                 debug=None):
        """Initialize."""
        self._func = func
        self._num_samples = num_samples
        self._num_elites = num_elites
        self._num_iterations = tf.constant(num_iterations)
        self._debug = debug

    def __call__(self, inputs, samples, seed, scores=None):
        """Returns an output using CEM."""
        def cond(i, samples, scores):
            return tf.less(i, self._num_iterations)

        def body(i, samples, scores):
            """Defines the body of the while loop.
            
            Args:
                i: The iteration index as a tensor.
                samples: A tensor of shape [num_samples, ...].
                scores: A tensor of shape [num_samples, ...].

            Returns: 
                Tuple of i + 1, new_samples, new_scores
            """
            _, ind = tf.nn.top_k(scores, self._num_elites)
            elites = tf.gather(samples, ind)
            mean, var = tf.nn.moments(elites, axes=0)

            # if self._debug:
            #     print('samples', samples.get_shape())
            #     print('scores', scores.get_shape())
            #     print('ind', ind.get_shape())
            #     print('elites', elites)

            new_dist = tf.distributions.Normal(loc=mean, scale=tf.sqrt(var))
            new_samples = new_dist.sample([self._num_samples])
            new_scores = self._func(inputs, new_samples)

            # Print debugging information.
            if self._debug:
                print_op = tf.print(
                    '-- CEM Iter ', i, ' --\n',
                    'samples: ', samples, '\n',
                    'scores: ', scores, '\n',
                    'elite_indices: ', ind, '\n',
                    'elite_samples: ', elites, '\n',
                    'elite_scores: ', tf.gather(scores, ind), '\n',
                )
                with tf.control_dependencies([print_op]):
                    i = tf.identity(i)

            return tf.add(i, 1), new_samples, new_scores

        if scores is None:
            scores = self._func(inputs, samples)

        if samples.shape[0] > self._num_samples:
            _, ind = tf.nn.top_k(scores, self._num_samples)
            samples = tf.gather(samples, ind)
            scores = tf.gather(scores, ind)

        _, samples, scores = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[0, samples, scores])

        _, ind = tf.nn.top_k(scores, k=1)
        top_sample = tf.gather(samples, ind)
        top_score = tf.gather(scores, ind)

        # Print debugging information.
        if self._debug:
            print_op = tf.print(
                '-- CEM Final --', '\n',
                'scores: ', scores, '\n',
                'top_indice: ', ind, '\n',
                'top_sample: ', top_sample, '\n',
                'top_score: ', top_score, '\n',
            )
            with tf.control_dependencies([print_op]):
                top_sample = tf.identity(top_sample)

        return top_sample
