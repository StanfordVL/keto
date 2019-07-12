"""Train and Eval DQN.
To run:
```bash
tf_agents/agents/dqn/examples/train_eval_gym \
 --root_dir=$HOME/tmp/dqn/gym/cart-pole/ \
 --alsologtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import flags

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.dqn import q_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common as common_utils
import gin.tf

nest = tf.contrib.framework.nest

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'), 
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
        root_dir,
        env_name='CartPole-v0',
        num_iterations=100000,
        # Params for collect
        initial_collect_steps=1000,
        collect_steps_per_iteration=1,
        epsilon_greedy=0.1,
        replay_buffer_capacity=100000,
        # Params for target update
        target_update_tau=0.05,
        target_update_period=5,
        # Params for train
        train_steps_per_iteration=1,
        batch_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        # Params for eval
        num_eval_episodes=10,
        eval_interval=1000,
        # Params for checkpoints, summaries, and logging
        train_checkpoint_interval=10000,
        policy_checkpoint_interval=5000,
        rb_checkpoint_interval=20000,
        log_interval=1000,
        summary_interval=1000,
        summaries_flush_secs=10,
        agent_class=dqn_agent.DqnAgent,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        eval_metrics_callback=None):
    """A simple train and eval for DQN."""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.contrib.summary.create_file_writer(
            train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.contrib.summary.create_file_writer(
            eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        py_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        py_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
    ]

    # TODO(kbanoop): Figure out if it is possible to avoid the with block.
    with tf.contrib.summary.record_summaries_every_n_global_steps(
            summary_interval):

        tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
        eval_py_env = suite_gym.load(env_name)

        # TODO(kuanfang)
        q_net = q_network.QNetwork(
            tf_env.time_step_spec().observation,
            tf_env.action_spec(),
            fc_layer_params=None)

        tf_agent = agent_class(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
            epsilon_greedy=epsilon_greedy,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=dqn_agent.element_wise_squared_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars)

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec(),
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity)

        eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy())

        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]

        global_step = tf.train.get_or_create_global_step()

        replay_observer = [replay_buffer.add_batch]
        initial_collect_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(), tf_env.action_spec())
        initial_collect_op = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            initial_collect_policy,
            observers=replay_observer,
            num_steps=initial_collect_steps).run()

        collect_policy = tf_agent.collect_policy()
        collect_op = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=replay_observer + train_metrics,
            num_steps=collect_steps_per_iteration).run()

        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)

        iterator = dataset.make_initializable_iterator()
        trajectories, _ = iterator.get_next()
        train_op = tf_agent.train(
            experience=trajectories, train_step_counter=global_step)

        train_checkpointer = common_utils.Checkpointer(
            ckpt_dir=train_dir,
            agent=tf_agent,
            global_step=global_step,
            metrics=tf.contrib.checkpoint.List(train_metrics))
        policy_checkpointer = common_utils.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=tf_agent.policy(),
            global_step=global_step)
        rb_checkpointer = common_utils.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
            max_to_keep=1,
            replay_buffer=replay_buffer)

        for train_metric in train_metrics:
            train_metric.tf_summaries(step_metrics=train_metrics[:2])
        summary_op = tf.contrib.summary.all_summary_ops()

        with (eval_summary_writer.as_default(),
              tf.contrib.summary.always_record_summaries()):
            for eval_metric in eval_metrics:
                eval_metric.tf_summaries()

        init_agent_op = tf_agent.initialize()

        with tf.Session() as sess:
            # Initialize the graph.
            train_checkpointer.initialize_or_restore(sess)
            rb_checkpointer.initialize_or_restore(sess)
            sess.run(iterator.initializer)
            # TODO(sguada) Remove once Periodically can be saved.
            common_utils.initialize_uninitialized_variables(sess)

            sess.run(init_agent_op)
            tf.contrib.summary.initialize(session=sess)
            sess.run(initial_collect_op)

            global_step_val = sess.run(global_step)
            metric_utils.compute_summaries(
                    eval_metrics,
                    eval_py_env,
                    eval_py_policy,
                    num_episodes=num_eval_episodes,
                    global_step=global_step_val,
                    callback=eval_metrics_callback,
            )

            collect_call = sess.make_callable(collect_op)
            train_step_call = sess.make_callable(
                [train_op, summary_op, global_step])

            timed_at_step = sess.run(global_step)
            collect_time = 0
            train_time = 0
            steps_per_second_ph = tf.placeholder(
                    tf.float32, shape=(), name='steps_per_sec_ph')
            steps_per_second_summary = tf.contrib.summary.scalar(
                    name='global_steps/sec', tensor=steps_per_second_ph)

            for _ in range(num_iterations):
                # Train/collect/eval.
                start_time = time.time()
                collect_call()
                collect_time += time.time() - start_time
                start_time = time.time()
                for _ in range(train_steps_per_iteration):
                    loss_info_value, _, global_step_val = train_step_call()
                train_time += time.time() - start_time

                if global_step_val % log_interval == 0:
                    tf.logging.info(
                        'step = %d, loss = %f',
                        global_step_val, loss_info_value.loss)
                    steps_per_sec = (
                        (global_step_val - timed_at_step) /
                        (collect_time + train_time))
                    sess.run(
                            steps_per_second_summary,
                            feed_dict={steps_per_second_ph: steps_per_sec})
                    tf.logging.info('%.3f steps/sec' % steps_per_sec)
                    tf.logging.info('collect_time = {}, train_time = {}'.format(
                            collect_time, train_time))
                    timed_at_step = global_step_val
                    collect_time = 0
                    train_time = 0

                if global_step_val % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step_val)

                if global_step_val % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step_val)

                if global_step_val % rb_checkpoint_interval == 0:
                    rb_checkpointer.save(global_step=global_step_val)

                if global_step_val % eval_interval == 0:
                    metric_utils.compute_summaries(
                            eval_metrics,
                            eval_py_env,
                            eval_py_policy,
                            num_episodes=num_eval_episodes,
                            global_step=global_step_val,
                            callback=eval_metrics_callback,
                    )


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    agent_class = dqn_agent.DdqnAgent
    train_eval(
        FLAGS.root_dir,
        agent_class=agent_class,
        num_iterations=FLAGS.num_iterations)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    tf.app.run()
