import tensorflow as tf
import argparse

from cvae.build import build_inference_graph

parser = argparse.ArgumentParser()
parser.add_argument('--vae', 
                    type=str,
                    required=True)
parser.add_argument('--gcnn',
                    type=str,
                    required=True)
parser.add_argument('--output',
                    type=str,
                    default='./runs/cvae_model')
args = parser.parse_args()

graph_inf = build_inference_graph()

with tf.Session() as sess:

    vars = tf.global_variables()
    vars_vae = [var for var in vars if 'vae' in var.name]
    vars_gcnn = [var for var in vars if 'discr' in var.name]

    saver = tf.train.Saver(var_list=vars)
    saver_vae = tf.train.Saver(var_list=vars_vae)
    saver_gcnn = tf.train.Saver(var_list=vars_gcnn)

    saver_vae.restore(sess, args.vae)
    saver_gcnn.restore(sess, args.gcnn)
    saver.save(sess, args.output)


