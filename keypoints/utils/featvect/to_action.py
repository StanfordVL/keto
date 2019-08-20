import h5py
import argparse
import numpy as np
import tensorflow as tf

from keypoints_to_action import to_action_hammer
from keypoints_to_action import to_action_push
from keypoints_to_action import to_action_reach

parser = argparse.ArgumentParser()

parser.add_argument('--task_name',
                    type=str,
                    required=True,
                    help='hammer, push or reach')

parser.add_argument('--input',
                    type=str,
                    required=True)

parser.add_argument('--output',
                    type=str,
                    default='./data_action.hdf5')

args = parser.parse_args()

point_cloud_tf = tf.placeholder(tf.float32, [1024, 3])
grasp_point_tf = tf.placeholder(tf.float32, [3])
funct_point_tf = tf.placeholder(tf.float32, [3])

if args.task_name == 'hammer':
    num_keypoints = 2
    keypoints = [grasp_point_tf, funct_point_tf]
    c_xy, c_rz = to_action_hammer(point_cloud_tf, keypoints)

elif args.task_name == 'push':
    num_keypoints = 3
    funct_vect_tf = tf.placeholder(tf.float32, [3])
    keypoints = [grasp_point_tf, funct_point_tf, funct_vect_tf]
    c_xy, c_rz = to_action_push(point_cloud_tf, keypoints)

elif args.task_name == 'reach':
    num_keypoints = 3
    funct_vect_tf = tf.placeholder(tf.float32, [3])
    keypoints = [grasp_point_tf, funct_point_tf, funct_vect_tf]
    c_xy, c_rz = to_action_reach(point_cloud_tf, keypoints)

else:
    raise ValueError


f_input = h5py.File(args.input, 'r')
f_output = h5py.File(args.output, 'w')

pos_point_cloud = f_input['pos_point_cloud']
neg_point_cloud = f_input['neg_point_cloud']

pos_keypoints = f_input['pos_keypoints']
neg_keypoints = f_input['neg_keypoints']

num_pos = pos_point_cloud.shape[0]
num_neg = neg_point_cloud.shape[0]

f_output.create_dataset('pos_point_cloud', shape=[num_pos, 1024, 3])
f_output.create_dataset('pos_action', shape=[num_pos, 3])

f_output.create_dataset('neg_point_cloud', shape=[num_neg, 1024, 3])
f_output.create_dataset('neg_action', shape=[num_neg, 3])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:

    for index in range(num_pos):
        pc = pos_point_cloud[index]
        g_kp_np, f_kp_np, f_v_np = np.split(
                np.squeeze(pos_keypoints[index]), [1, 2], axis=0)
    
        if args.task_name == 'hammer':
            c_xy_np, c_rz_np = sess.run([c_xy, c_rz], 
                    feed_dict={point_cloud_tf: pc,
                               grasp_point_tf: np.squeeze(g_kp_np),
                               funct_point_tf: np.squeeze(f_kp_np)})
            action = np.array([c_xy_np[0], c_xy_np[1], c_rz_np],
                    dtype=np.float32)
            f_output['pos_point_cloud'][index] = pc
            f_output['pos_action'][index] = action

        elif args.task_name in ['push', 'reach']:
            c_xy_np, c_rz_np = sess.run([c_xy, c_rz], 
                    feed_dict={point_cloud_tf: pc,
                               grasp_point_tf: np.squeeze(g_kp_np),
                               funct_point_tf: np.squeeze(f_kp_np),
                               funct_vect_tf: np.squeeze(f_v_np)})
            action = np.array([c_xy_np[0], c_xy_np[1], c_rz_np],
                    dtype=np.float32)
            f_output['pos_point_cloud'][index] = pc
            f_output['pos_action'][index] = action
        else:
            raise NotImplementedError

    for index in range(num_neg):
        pc = neg_point_cloud[index]
        g_kp_np, f_kp_np, f_v_np = np.split(
                np.squeeze(neg_keypoints[index]), [1, 2], axis=0)
    
        if args.task_name == 'hammer':
            c_xy_np, c_rz_np = sess.run([c_xy, c_rz], 
                    feed_dict={point_cloud_tf: pc,
                               grasp_point_tf: np.squeeze(g_kp_np),
                               funct_point_tf: np.squeeze(f_kp_np)})
            action = np.array([c_xy_np[0], c_xy_np[1], c_rz_np],
                    dtype=np.float32)
            f_output['neg_point_cloud'][index] = pc
            f_output['neg_action'][index] = action

        elif args.task_name in ['push', 'reach']:
            c_xy_np, c_rz_np = sess.run([c_xy, c_rz], 
                    feed_dict={point_cloud_tf: pc,
                               grasp_point_tf: np.squeeze(g_kp_np),
                               funct_point_tf: np.squeeze(f_kp_np),
                               funct_vect_tf: np.squeeze(f_v_np)})
            action = np.array([c_xy_np[0], c_xy_np[1], c_rz_np],
                    dtype=np.float32)
            f_output['neg_point_cloud'][index] = pc
            f_output['neg_action'][index] = action
        else:
            raise NotImplementedError
