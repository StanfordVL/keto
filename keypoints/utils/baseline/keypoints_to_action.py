import numpy as np
import tensorflow as tf

from solver_quadratic import solver_quadratic


def rotate(x, rot):
    x = tf.concat(
            [[tf.cos(rot) * x[0] - tf.sin(rot) * x[1]],
             [tf.sin(rot) * x[0] + tf.cos(rot) * x[1]]],
            axis=0)
    return x


def to_action_hammer(point_cloud_tf,
                     keypoints,
                     target_pose=[0, 0, 0, 0]):

    center = tf.reduce_mean(
            tf.squeeze(point_cloud_tf), axis=0)
    g_kp, f_kp = [tf.squeeze(k) for k in keypoints]

    v_cf = tf.reshape(f_kp - center, [3])
    v_cg = tf.reshape(g_kp - center, [3])
    s = tf.sign(v_cg[0] * v_cf[1] -
                v_cg[1] * v_cf[0])

    theta = s * np.pi / 2

    action_xy = g_kp[:2]
    v_af = -action_xy + f_kp[:2]
    d = tf.linalg.norm(v_af)

    start_rz = tf.atan2(y=v_af[1], x=v_af[0])
    tx, ty, tz, trz = target_pose

    target = tf.constant([tx, ty], dtype=tf.float32)
    force = tf.constant([np.cos(trz), np.sin(trz)],
                        dtype=tf.float32)

    g_xy, g_rz = tf.py_func(solver_quadratic,
                            [target, force * 0.01, theta, d],
                            [tf.float32, tf.float32])
    g_rz = g_rz - start_rz
    
    return g_kp, g_xy, g_rz


def to_action_push(point_cloud_tf,
                   keypoints,
                   target_pose=[0, 0, 0, 0]):

    g_kp, f_kp, f_v = [tf.squeeze(k) for k in keypoints]
    v_fg = g_kp - f_kp
    theta = tf.add(tf.atan2(f_v[1], f_v[0]),
                   -tf.atan2(v_fg[1], v_fg[0]))

    action_xy = g_kp[:2]
    v_af = -action_xy + f_kp[:2]
    d = tf.linalg.norm(v_af)

    start_rz = tf.atan2(y=v_af[1], x=v_af[0])
    tx, ty, tz, trz = target_pose

    target = tf.constant([tx, ty], dtype=tf.float32)
    force = tf.constant([np.cos(trz), np.sin(trz)],
                        dtype=tf.float32)

    g_xy, g_rz = tf.py_func(solver_quadratic,
                            [target, force * 0.01, theta, d],
                            [tf.float32, tf.float32])
    g_rz = g_rz - start_rz
    
    return g_kp, g_xy, g_rz


def to_action_reach(point_cloud_tf,
                    keypoints,
                    target_pose=[0, 0, 0, np.pi/2]):

    g_kp, f_kp, f_v = [tf.squeeze(k) for k in keypoints]
    v_fg = g_kp - f_kp
    theta = tf.add(tf.atan2(f_v[1], f_v[0]),
                   -tf.atan2(v_fg[1], v_fg[0]))

    action_xy = g_kp[:2]
    v_af = -action_xy + f_kp[:2]
    d = tf.linalg.norm(v_af)

    start_rz = tf.atan2(y=v_af[1], x=v_af[0])
    tx, ty, tz, trz = target_pose

    target = tf.constant([tx, ty], dtype=tf.float32)
    force = tf.constant([np.cos(trz), np.sin(trz)],
                        dtype=tf.float32)

    g_xy, g_rz = tf.py_func(solver_quadratic,
                            [target, force * 0.01, theta, d],
                            [tf.float32, tf.float32])
    g_rz = g_rz - start_rz
    
    return g_kp, g_xy, g_rz
