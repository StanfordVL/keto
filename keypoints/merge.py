"""Merges the keypoints model and the grasping model."""
import tensorflow as tf
import argparse

from cvae.build import build_grasp_inference_graph
from cvae.build import build_keypoint_inference_graph
from cvae.build import build_action_inference_graph


parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    type=str,
                    required=True)
parser.add_argument('--vae', 
                    type=str)
parser.add_argument('--discr',
                    type=str)
parser.add_argument('--grasp',
                    type=str)
parser.add_argument('--keypoint',
                    type=str)
parser.add_argument('--action',
                    type=str)
parser.add_argument('--num_funct_vect',
                    type=str,
                    default='1')
parser.add_argument('--output',
                    type=str,
                    default='./runs/cvae_model')
args = parser.parse_args()

if args.model == 'grasp':
    build_grasp_inference_graph()
elif args.model == 'keypoint':
    build_keypoint_inference_graph(
            num_funct_vect=int(args.num_funct_vect))
elif args.model == 'grasp_keypoint':
    build_grasp_inference_graph()
    build_keypoint_inference_graph(
            num_funct_vect=int(args.num_funct_vect))
elif args.model == 'action':
    build_action_inference_graph()
elif args.model == 'grasp_action':
    build_grasp_inference_graph()
    build_action_inference_graph()

else:
    raise ValueError(args.model)

with tf.Session() as sess:

    vars = tf.global_variables()

    if args.model in ['grasp', 'keypoint', 'action']:
        # Merges the generation network (VAE) and 
        # the evaluation network (binary classifier).
        vars_vae = [var for var in vars if 'vae' in var.name]
        vars_discr = [var for var in vars if 'discr' in var.name]

        saver = tf.train.Saver(var_list=vars)
        saver_vae = tf.train.Saver(var_list=vars_vae)
        saver_discr = tf.train.Saver(var_list=vars_discr)
 
        saver_vae.restore(sess, args.vae)
        saver_discr.restore(sess, args.discr)
        saver.save(sess, args.output)

    elif args.model == 'grasp_keypoint':
        # Merges the grasp prediction network and the keypoints network
        vars_grasp = [var for var in vars if 'grasp' in var.name]
        vars_keypoint = [var for var in vars if 'keypoint' in var.name]

        saver = tf.train.Saver(var_list=vars)
        saver_grasp = tf.train.Saver(var_list=vars_grasp)
        saver_keypoint = tf.train.Saver(var_list=vars_keypoint)

        saver_grasp.restore(sess, args.grasp)
        saver_keypoint.restore(sess, args.keypoint)
        saver.save(sess, args.output)

    elif args.model == 'grasp_action':
        # Merges the grasp prediction network and the action network
        # This is only for the End-to-End baseline where we directly
        # predict the actions from the visual observation.
        vars_grasp = [var for var in vars if 'grasp' in var.name]
        vars_action = [var for var in vars if 'action' in var.name]

        saver = tf.train.Saver(var_list=vars)
        saver_grasp = tf.train.Saver(var_list=vars_grasp)
        saver_action = tf.train.Saver(var_list=vars_action)

        saver_grasp.restore(sess, args.grasp)
        saver_action.restore(sess, args.action)
        saver.save(sess, args.output)

    else:
        raise ValueError
