import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')

import utils
from cvae.discriminator import KeypointDiscriminator

# Hammering

part_1 = tf.placeholder(tf.float32, [1024, 3])
part_2 = tf.placeholder(tf.float32, [1024, 3])

trans_1 = tf.placeholder(tf.float32, [3])
trans_2 = tf.placeholder(tf.float32, [3])

rot_1 = tf.placeholder(tf.float32, ())
rot_2 = tf.placeholder(tf.float32, ())

part_1_move = tf.matmul(part_1, tf.transpose(
    utils.rotation_mat(rot_1), [1, 0])) + trans_1
part_2_move = tf.matmul(part_2, tf.transpose(
    utils.rotation_mat(rot_2), [1, 0])) + trans_2

point_cloud = tf.concat([part_1_move, part_2_move], axis=0)
grasp_point = tf.placeholder(tf.float32, [3])
funct_point = tf.placeholder(tf.float32, [3])
funct_vect = tf.placeholder(tf.float32, [3])

score = KeypointDiscriminator().build_model(
    tf.reshape(point_cloud, [1, 2048, 1, 3]),
    [tf.reshape(grasp_point, [1, 1, 3]), tf.reshape(funct_point, [1, 1, 3])],
    tf.reshape(funct_vect, [1, 1, 3]))

score = tf.squeeze(score)

dist = tf.add_n([utils.min_dist(part_1_move, grasp_point),
                 utils.min_dist(part_2_move, funct_point),
                 utils.min_dist(part_1_move, part_2_move)])
dist = dist * 100

trans_1_grad = tf.gradients(score - dist, trans_1)
trans_2_grad = tf.gradients(score - dist, trans_2)
rot_1_grad = tf.gradients(score - dist, rot_1)
rot_2_grad = tf.gradients(score - dist, rot_2)

part_1_path = '../../data/part/handle/point_cloud.npy'
part_2_path = '../../data/part/head/point_cloud.npy'
os.mkdir('visualize') if not os.path.exists('visualize') else None
scale = 20

part_1_np = np.load(open(part_1_path, 'rb')) * scale
part_2_np = np.load(open(part_2_path, 'rb')) * scale


with tf.Session() as sess:
    max_iter = 20
    learning_rate = 1e-3
    grasp_point_np = np.array([0, 0, 0], dtype=np.float32)
    funct_point_np = np.array([-4, 0, 0], dtype=np.float32)
    funct_vect_np = np.array([-1, 0, 0], dtype=np.float32)

    trans_1_np = np.array([0, 0, 0], dtype=np.float32)
    trans_2_np = np.array([-4, 0, 0], dtype=np.float32)
    rot_1_np = np.random.uniform(0, 3.14)
    rot_2_np = np.random.uniform(0, 3.14)

    saver = tf.train.Saver()
    saver.restore(sess, '../models/cvae_push')

    for iiter in range(max_iter):
        [trans_1_grad_np, trans_2_grad_np,
         rot_1_grad_np, rot_2_grad_np,
         part_1_move_np, part_2_move_np, score_np] = sess.run(
            [trans_1_grad, trans_2_grad, rot_1_grad, rot_2_grad,
             part_1_move, part_2_move, score],
            feed_dict={part_1: part_1_np, part_2: part_2_np,
                       trans_1: trans_1_np, trans_2: trans_2_np,
                       rot_1: rot_1_np, rot_2: rot_2_np,
                       grasp_point: grasp_point_np, funct_point: funct_point_np,
                       funct_vect: funct_vect_np})

        trans_1_np = trans_1_np + learning_rate * \
            trans_1_grad_np[0] * np.array([1, 1, 0])
        trans_2_np = trans_2_np + learning_rate * \
            trans_2_grad_np[0] * np.array([1, 1, 0])
        rot_1_np = rot_1_np + learning_rate * rot_1_grad_np[0] * 3
        rot_2_np = rot_2_np + learning_rate * rot_2_grad_np[0] * 3

        point_cloud_curr = np.concatenate(
            [part_1_move_np, part_2_move_np], axis=0)
        keypoints_list = [np.reshape(grasp_point_np, [1, 3]),
                          np.reshape(funct_point_np, [1, 3]),
                          np.reshape(funct_vect_np, [1, 3])]
        print('--------Step {}--------'.format(iiter))
        print('Handle trans: {}, rot: {}'.format(trans_1_np/scale, rot_1_np))
        print('Head trans: {}, rot: {}'.format(trans_2_np/scale, rot_2_np))
        utils.visualize_keypoints(point_cloud_curr, keypoints_list,
                                  prefix='visualize', name='%06d' % (iiter))
