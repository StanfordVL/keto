"""Builds the grasp and keypoint prediction network."""
import logging
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

from cvae.reader import GraspReader, KeypointReader
from cvae.encoder import GraspEncoder, KeypointEncoder
from cvae.decoder import GraspDecoder, KeypointDecoder
from cvae.discriminator import GraspDiscriminator, KeypointDiscriminator

import matplotlib as mpl
mpl.use('Agg')

logging.basicConfig(level=logging.DEBUG)


def makedir(path):
    if not os.path.exists(path):
        os.mkdirs(path)
    return


class RunningLog(object):
    """Logging utils."""

    def __init__(self, filename):
        self.filename = filename
        return

    def write(self, ext, s):
        with open(self.filename + '.' + ext, 'a') as f:
            f.write(s + '\r\n')
            print(s)
        return


running_log = RunningLog('./runs/running_log')


def rotation_matrix(alpha, beta, gamma):
    """Computes the rotation matrix.
    
    Args:
        alpha: The rotation around x axis.
        beta: The rotation around y axis.
        gamma: The rotation around z axis.
        
    Return:
        R: The rotation matrix.
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    R = np.dot(np.dot(Rz, Ry), Rx)
    return R


def visualize(point_cloud, pose_rot, prefix, name,
              plot_lim=2, point_size=0.05):
    """Visualizes the grasps on point cloud.
    
    Args:
        point_cloud: A numpy array of the point cloud.
        pose_rot: The location and rotation of the grasps.
        prefix: The directory to save the figure.
        name: The name of the figure.
        plot_lim: The range of x and y axis of the plot.
        point_size: The size of points when drawing the point cloud.
        
    Return:
        None.
    """
    fig = plt.figure(figsize=(10, 6))
    plot_num = 111
    xs = point_cloud[:, 0]
    ys = point_cloud[:, 1]
    zs = point_cloud[:, 2]
    ax = fig.add_subplot(plot_num, projection='3d')
    ax.view_init(elev=90, azim=0)
    ax.scatter(xs, ys, zs, s=point_size)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_axis_off()
    ax.grid(False)

    for grasp_num in range(pose_rot.shape[0]):
        grasp_pos = pose_rot[grasp_num, 0:3]
        grasp_ori = pose_rot[grasp_num, 3:6]
        alpha = grasp_ori[0]
        beta = grasp_ori[1]
        gamma = grasp_ori[2]
        R = rotation_matrix(alpha, beta, gamma)
        grasp_dir = np.reshape([0, 0.1, 0], [3, 1])
        grasp_pos = np.reshape(grasp_pos, [3, 1])
        grasp_dir = np.dot(R, grasp_dir) + grasp_pos
        grasp_dir = np.concatenate([grasp_dir, grasp_pos], 1)
        ax.plot(grasp_dir[0], grasp_dir[1],
                grasp_dir[2], c='green')
        ax.scatter(grasp_pos[0], grasp_pos[1],
                   grasp_pos[2], c='red', s=point_size)

    plt.savefig(os.path.join(prefix, '{}.png'.format(name)))
    plt.close()
    return


def visualize_keypoints(point_cloud,
                        keypoints,
                        prefix,
                        name,
                        plot_lim=2,
                        point_size=0.5):
    """Visualizes the keypoints on point cloud.
    
    Args:
        point_cloud: A numpy array of the point cloud.
        keypoints: A list of keypoints [grasp point, function point].
        prefix: The directory to save the figure.
        name: The name of the figure.
        plot_lim: The range of x and y axis of the plot.
        point_size: The size of points when drawing the point cloud.
        
    Return:
        None.
    """
    [grasp_point, funct_point] = keypoints
    fig = plt.figure(figsize=(18, 6))
    xs = point_cloud[:, 0]
    ys = point_cloud[:, 1]
    zs = point_cloud[:, 2]

    ax = fig.add_subplot(131, projection='3d')
    ax.view_init(elev=90, azim=0)
    ax.scatter(xs, ys, zs, s=point_size, c='grey')
    ax.set_axis_off()
    ax.grid(False)
    ax.scatter(grasp_point[:, 0],
               grasp_point[:, 1],
               grasp_point[:, 2],
               s=point_size * 40,
               c='darkgreen')

    ax = fig.add_subplot(132, projection='3d')
    ax.view_init(elev=90, azim=0)
    ax.scatter(xs, ys, zs, s=point_size, c='silver')
    ax.set_axis_off()
    ax.grid(False)
    ax.scatter(funct_point[:, 0],
               funct_point[:, 1],
               funct_point[:, 2],
               s=point_size * 40,
               c='darkred')

    plt.savefig(os.path.join(prefix, '{}.png'.format(name)))
    plt.close()
    return


def rectify_keypoints(point_cloud,
                      grasp_point,
                      funct_point,
                      funct_on_hull,
                      grasp_clusters=12,
                      funct_clusters=12):
    """Replaces the grasp point and the function point with
       the nearest clustering center of the point cloud.
       
    Args:
        point_cloud: A numpy array of point cloud.
        grasp_point: The grasp point.
        funct_point: The function point.
        funct_on_hull: A bool indicating whether the 
            function point should be on the convex 
            hull of the clustering centers.
        grasp_clusters: The number of clusters for 
            rectifying the grasp point.
        funct_clusters: The number of clusters for 
            rectifying the function point.
        
    Returns:
        grasp_point: The rectified grasp point.
        funct_point: The rectified function point.
    """
       
    p = np.squeeze(point_cloud)
    kmeans = KMeans(
        n_clusters=grasp_clusters,
        random_state=0).fit(p)
    centers = kmeans.cluster_centers_
    index = np.argsort(
        np.linalg.norm(grasp_point - centers, axis=1))[0]
    grasp_point = centers[np.newaxis, index].astype(np.float32)

    kmeans = KMeans(
        n_clusters=funct_clusters,
        random_state=0).fit(p)
    centers = kmeans.cluster_centers_

    if funct_on_hull:
        hull = ConvexHull(centers[:, :2])
        hull = centers[hull.vertices]
    else:
        hull = centers

    hull_index = np.argsort(
        np.linalg.norm(funct_point - hull, axis=1))[0]
    funct_point = hull[np.newaxis, hull_index].astype(np.float32)

    return grasp_point, funct_point


def success_recall(pose_pred, pose_ref,
                   loc_thres=0.1, cos_thres=0.9):
    """Measures if a groud truth grasp is recalled.
    
    Args: 
        pose_pred: The predicted grasp.
        pose_ref: The ground truth grasp.
        loc_thres: The Euler distance threshold to be satisfied.
        cos_thres: The cosine distance threshold to be satisfied.
        
    Returns:
        success: A bool indicating whether the ground truth is recalled.
    """
    diff = pose_pred - pose_ref
    diff_loc = diff[:, :3]
    diff_rot = diff[:, 3:]
    diff_loc_norm = np.linalg.norm(diff_loc, axis=1)
    diff_rot_min = np.amin(np.cos(diff_rot), axis=1)
    success = np.sum(np.logical_and(diff_loc_norm < loc_thres,
                                    diff_rot_min > cos_thres)) > 0
    return success


def centralize_point_cloud(point_cloud,
                           gripper_pose, norm=False):
    """Subtracts the mean point cloud from the point cloud and gripper pose.
        
    Args:
        point_cloud: A numpy array of point cloud.
        gripper_pose: A numpy array of the robot gripper pose.
        norm: Whether to normalize the coordinates.
        
    Returns:
        point_cloud_out: The centralized point cloud.
        gripper_pose_out: The centralized gripper pose.
    """
    center = np.mean(point_cloud, axis=1, keepdims=True)
    point_cloud_cent = point_cloud - center
    r = np.linalg.norm(point_cloud_cent, axis=2, keepdims=True)
    r_std = np.std(r, axis=1, keepdims=True)
    if norm:
        point_cloud_out = point_cloud_cent / (r_std + 1e-12)
        gripper_pose_out = np.concatenate([
            (gripper_pose[:, :3] - center) /
            (r_std[:, :, 0] + 1e-12),
            gripper_pose[:, 3:]], axis=1)
    else:
        point_cloud_out = point_cloud_cent
        gripper_pose_out = np.concatenate([
            gripper_pose[:, :3] - center,
            gripper_pose[:, 3:]], axis=1)
    return point_cloud_out, gripper_pose_out


def pose_rot_to_vect(gripper_pose):
    """Replaces the rotation with cosine and sine.
        
    Args:
        gripper_pose: A tensor of gripper pose.
       
    Returns:
        rot_vect: The gripper pose whose rotations 
            are replaced with cosine and sine.
    """
    loc, rot = tf.split(gripper_pose, [3, 3], axis=1)
    rx, ry, rz = tf.split(rot, [1, 1, 1], axis=1)
    rot_vect = tf.concat([loc, tf.cos(rx), tf.sin(rx),
                          tf.cos(ry), tf.sin(ry),
                          tf.cos(rz), tf.sin(rz)],
                         axis=1)
    return rot_vect


def pose_vect_to_rot(gripper_pose):
    """Replaces the cosine and sine with the rotation.
        
    Args:
        gripper_pose: A tensor of gripper pose.
       
    Returns:
        pose_rot: The gripper pose where the cosine and
            sine are replaced with rotation.
    """
    [loc, rx_cos, rx_sin, ry_cos, ry_sin,
        rz_cos, rz_sin] = tf.split(gripper_pose,
                                   [3, 1, 1, 1, 1, 1, 1], axis=1)
    rx = tf.atan2(rx_sin, rx_cos)
    ry = tf.atan2(ry_sin, ry_cos)
    rz = tf.atan2(rz_sin, rz_cos)
    pose_rot = tf.concat([loc, rx, ry, rz], axis=1)
    return pose_rot


def reduce_std(x, axis=-1,
               keepdims=False):
    """Computes the standard deviation.
    
    Args:
        x: The input tensor.
        axis: The axis on which we compute STD.
        keepdims: If true, the axis will be kept.
    
    Return:
        The standard deviation.
    """
    x_c = x - tf.reduce_mean(x,
                             axis=axis, keepdims=True)
    std = tf.sqrt(tf.reduce_mean(
        tf.square(x_c), axis=axis,
        keepdims=keepdims))
    return std


def build_grasp_training_graph(num_points=1024):
    """Builds the tensorflow graph for training the grasp model.
    
    Args: 
        num_points: The number of points in one frame of point cloud.
        
    Return:
        The training graph for the grasp model.
    """
    point_cloud_tf = tf.placeholder(dtype=tf.float32,
                                    shape=[None, num_points, 3])
    pose_tf = tf.placeholder(dtype=tf.float32,
                             shape=[None, 6])
    pose_label_tf = tf.placeholder(dtype=tf.float32,
                                   shape=[None, 1])
    pose_vect = pose_rot_to_vect(pose_tf)
    pose_neg_tf = tf.placeholder(dtype=tf.float32,
                                 shape=[None, 6])
    latent_var = GraspEncoder().build_model(tf.reshape(point_cloud_tf,
                                                       (-1, num_points, 1, 3)),
                                            pose_vect)
    z_mean, z_std = tf.split(latent_var, 2, axis=1)
    z = z_mean + z_std * tf.random.normal(tf.shape(z_std))
    z_std = tf.reduce_mean(reduce_std(z, axis=0))
    z_mean = tf.reduce_mean(z_mean)

    pose_vae_out = GraspDecoder().build_model(tf.reshape(
        point_cloud_tf, (-1, num_points, 1, 3)), latent_var)

    pose_vect_expand = tf.expand_dims(pose_vect, 2)

    loss_vae_loc = tf.reduce_mean(
        tf.abs(pose_vect_expand[:, :3] - pose_vae_out[:, :3]),
        axis=1, keepdims=True)

    loss_vae_rot = tf.reduce_mean(
        tf.abs(pose_vect_expand[:, 3:] - pose_vae_out[:, 3:]),
        axis=1, keepdims=True)

    loss_sum = loss_vae_loc
    mask = tf.equal(loss_sum,
                    tf.reduce_min(loss_sum, axis=2, keepdims=True))
    mask = tf.cast(mask, tf.float32)

    loss_vae_loc = tf.reduce_mean(loss_vae_loc * mask)
    loss_vae_rot = tf.reduce_mean(loss_vae_rot * mask)

    pose_vae_out_loc = tf.expand_dims(
        pose_vae_out[:, :3], axis=1)
    point_cloud_expand = tf.expand_dims(point_cloud_tf, 3)
    dist = point_cloud_expand - pose_vae_out_loc
    dist = tf.reduce_mean(dist * dist, axis=2)
    dist = tf.reduce_min(dist, axis=1)
    loss_vae_dist = tf.reduce_mean(dist)

    point_cloud_mean = tf.reduce_mean(
        point_cloud_tf, axis=1)
    std_gt_loc = tf.reduce_mean(
        reduce_std(pose_vect[:, :3] -
                   point_cloud_mean, axis=1))
    std_gt_rot = tf.reduce_mean(
        reduce_std(pose_vect[:, 3:], axis=1))

    miu, sigma = tf.split(latent_var, 2, axis=1)
    loss_vae_mmd = tf.reduce_mean(tf.square(miu) +
                                  tf.square(sigma) -
                                  tf.log(1e-8 + tf.square(sigma)) - 1)

    discr_pos, aligned_point_cloud = \
        GraspDiscriminator().build_model(tf.reshape(point_cloud_tf,
                                                    (-1, num_points, 1, 3)),
                                         pose_tf, aligned_point_cloud=True)
    discr_neg = GraspDiscriminator().build_model(tf.reshape(
        point_cloud_tf, (-1, num_points, 1, 3)), pose_neg_tf)
    loss_discr_pos = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=pose_label_tf, logits=discr_pos)
    loss_discr_pos = tf.reduce_mean(loss_discr_pos)
    loss_discr_neg = -tf.reduce_mean(
        tf.log(1.0 - tf.sigmoid(discr_neg)))
    pred_label = tf.cast(tf.greater(tf.sigmoid(discr_pos), 0.5),
                         tf.float32)
    pred_equal_gt = tf.cast(tf.equal(pred_label,
                                     pose_label_tf), tf.float32)

    acc_discr = tf.reduce_mean(pred_equal_gt)

    prec_discr = tf.math.divide(
        tf.reduce_sum(pred_equal_gt * pred_label),
        tf.reduce_sum(pred_label) + 1e-6)

    training_graph = {'point_cloud_tf': point_cloud_tf,
                      'pose_tf': pose_tf,
                      'pose_label_tf': pose_label_tf,
                      'pose_neg_tf': pose_neg_tf,
                      'pose_rot': pose_vect_to_rot(pose_vae_out),
                      'loss_vae_loc': loss_vae_loc,
                      'loss_vae_rot': loss_vae_rot,
                      'loss_vae_mmd': loss_vae_mmd,
                      'loss_vae_dist': loss_vae_dist,
                      'z_mean': z_mean,
                      'z_std': z_std,
                      'std_gt_loc': std_gt_loc,
                      'std_gt_rot': std_gt_rot,
                      'loss_discr_pos': loss_discr_pos,
                      'loss_discr_neg': loss_discr_neg,
                      'acc_discr': acc_discr,
                      'prec_discr': prec_discr,
                      'aligned_point_cloud': aligned_point_cloud}
    return training_graph


def build_keypoint_training_graph(num_points=1024,
                                  num_grasp_point=1,
                                  num_funct_point=1,
                                  num_funct_vect=1):
    """Builds the tensorflow graph for training the keypoint model.
    
    Args: 
        num_points: The number of points in one frame of point cloud.
        num_grasp_point: The number of grasp point.
        num_funct_point: The number of function point.
        num_funct_vect: The number of vectors pointing from the 
            function point to the effect point.

    Return:
        The training graph for the keypoint prediction model.
    """
    point_cloud_tf = tf.placeholder(dtype=tf.float32,
                                    shape=[None, num_points, 3])
    grasp_point_tf = tf.placeholder(dtype=tf.float32,
                                    shape=[None, num_grasp_point, 3])
    funct_point_tf = tf.placeholder(dtype=tf.float32,
                                    shape=[None, num_funct_point, 3])
    funct_vect_tf = None
    if num_funct_vect:
        funct_vect_tf = tf.placeholder(dtype=tf.float32,
                                       shape=[None, num_funct_vect, 3])

    keypoints = [grasp_point_tf, funct_point_tf]
    keypoints_label_tf = tf.placeholder(dtype=tf.float32,
                                        shape=[None, 1])
    latent_var = KeypointEncoder().build_model(tf.reshape(
        point_cloud_tf, (-1, num_points, 1, 3)), keypoints, funct_vect_tf)
    z_mean, z_std = tf.split(latent_var, 2, axis=1)
    z = z_mean + z_std * tf.random.normal(tf.shape(z_std))
    z_std = tf.reduce_mean(reduce_std(z, axis=1))
    z_mean = tf.reduce_mean(z_mean)

    keypoints_vae, funct_vect_vae = KeypointDecoder().build_model(tf.reshape(
        point_cloud_tf, (-1, num_points, 1, 3)), latent_var, num_funct_vect)

    [grasp_point_vae, funct_point_vae] = keypoints_vae

    loss_vae_grasp = tf.reduce_mean(
        tf.abs(grasp_point_vae - grasp_point_tf))

    loss_vae_funct = tf.reduce_mean(
        tf.abs(funct_point_vae - funct_point_tf))

    loss_vae_funct_vect = tf.constant(0.0, dtype=tf.float32)
    if num_funct_vect:
        loss_vae_funct_vect = tf.reduce_mean(
            tf.abs(funct_vect_vae - funct_vect_tf))

    point_cloud_mean = tf.reduce_mean(
        point_cloud_tf, axis=1, keepdims=True)

    std_gt_grasp = tf.reduce_mean(
        reduce_std(grasp_point_tf -
                   point_cloud_mean, axis=0))
    std_gt_funct = tf.reduce_mean(reduce_std(
        funct_point_tf - point_cloud_mean, axis=0))

    std_gt_funct_vect = tf.constant(0.0, dtype=tf.float32)
    if num_funct_vect:
        std_gt_funct_vect = tf.reduce_mean(reduce_std(
            funct_vect_tf, axis=0))

    miu, sigma = tf.split(latent_var, 2, axis=1)
    loss_vae_mmd = tf.reduce_mean(tf.square(miu) +
                                  tf.square(sigma) -
                                  tf.log(1e-8 + tf.square(sigma)) - 1)

    discr_logit = KeypointDiscriminator().build_model(
        tf.reshape(point_cloud_tf, (-1, num_points, 1, 3)),
        keypoints, funct_vect_tf)

    loss_discr = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=keypoints_label_tf, logits=discr_logit)

    loss_discr = tf.reduce_mean(loss_discr)

    pred_label = tf.cast(tf.greater(tf.sigmoid(discr_logit), 0.5),
                         tf.float32)
    pred_equal_gt = tf.cast(tf.equal(pred_label,
                                     keypoints_label_tf), tf.float32)
    acc_discr = tf.reduce_mean(pred_equal_gt)

    prec_discr = tf.math.divide(
        tf.reduce_sum(pred_equal_gt * pred_label),
        tf.reduce_sum(pred_label) + 1e-6)

    training_graph = {'point_cloud_tf': point_cloud_tf,
                      'grasp_point_tf': grasp_point_tf,
                      'funct_point_tf': funct_point_tf,
                      'funct_vect_tf': funct_vect_tf,
                      'keypoints_label_tf': keypoints_label_tf,
                      'loss_vae_grasp': loss_vae_grasp,
                      'loss_vae_funct': loss_vae_funct,
                      'loss_vae_funct_vect': loss_vae_funct_vect,
                      'loss_vae_mmd': loss_vae_mmd,
                      'z_mean': z_mean,
                      'z_std': z_std,
                      'std_gt_grasp': std_gt_grasp,
                      'std_gt_funct': std_gt_funct,
                      'std_gt_funct_vect': std_gt_funct_vect,
                      'loss_discr': loss_discr,
                      'acc_discr': acc_discr,
                      'prec_discr': prec_discr}
    return training_graph


def get_learning_rate(step, steps, init=5e-4):
    """Adjusts the learning rate in training.
    
    Args: 
        step: The current step.
        steps: The total steps in training.
        init: The initial learning rate.
        
    Returns:
        The learning rate.
    """
    return init if step < steps * 0.8 else init * 0.1


def build_grasp_inference_graph(num_points=1024, num_samples=128):
    """Builds the inference graph for grasping.
    
    Args:
        num_points: The number of points in point cloud.
        num_samples: The number of samples from the grasp generator (VAE).
        
    Returns:
        The inference graph.
    """
    point_cloud_tf = tf.placeholder(dtype=tf.float32,
                                    shape=[1, num_points, 3])
    point_cloud = tf.tile(point_cloud_tf, [num_samples, 1, 1])
    latent_var = tf.concat([tf.zeros([num_samples, 2], dtype=tf.float32),
                            tf.ones([num_samples, 2], dtype=tf.float32)],
                           axis=1)
    pose_vae_out = GraspDecoder().build_model(tf.reshape(
        point_cloud, (-1, num_points, 1, 3)), latent_var)
    pose_rot = pose_vect_to_rot(pose_vae_out)
    pose_rot = tf.reshape(tf.transpose(pose_rot, [0, 2, 1]), (-1, 6))
    num_grasp = pose_rot.get_shape().as_list()[0]
    point_cloud_discr = tf.tile(point_cloud_tf, [num_grasp, 1, 1])
    score = GraspDiscriminator().build_model(
        tf.expand_dims(point_cloud_discr, 2), pose_rot)
    pose_loc = tf.expand_dims(pose_rot[:, :3], 1)
    dist = tf.linalg.norm(pose_loc - point_cloud_discr, axis=2)
    dist_min = tf.reduce_min(dist, axis=1)
    inference_graph = {'point_cloud_tf': point_cloud_tf,
                       'pose_rot': pose_rot,
                       'score': score,
                       'dist_min': dist_min}
    return inference_graph


def build_keypoint_inference_graph(num_points=1024,
                                   num_samples=128,
                                   num_funct_vect=0):
    """Builds the inference graph for keypoints prediction.
    
    Args:
        num_points: The number of points in point cloud.
        num_samples: The number of samples from the keypoint generator (VAE).
        num_funct_vect: The number of vectors pointing from the function 
            point to the effect point.
        
    Returns:
        The inference graph.
    """
    point_cloud_tf = tf.placeholder(
        tf.float32, [1, num_points, 3])
    point_cloud = tf.tile(
        point_cloud_tf, [num_samples, 1, 1])
    latent_var = tf.concat(
        [tf.zeros([num_samples, 2], dtype=tf.float32),
         tf.ones([num_samples, 2], dtype=tf.float32)],
        axis=1)
    keypoints_vae, funct_vect_vae = KeypointDecoder(
    ).build_model(tf.reshape(point_cloud,
                             (-1, num_points, 1, 3)),
                  latent_var, num_funct_vect)
    dist_mat = tf.linalg.norm(
        tf.add(tf.expand_dims(point_cloud, 3),
               -tf.transpose(
            tf.concat([tf.expand_dims(k, 3) for
                       k in keypoints_vae], axis=1),
            [0, 3, 2, 1])), axis=2)
    dist = tf.reduce_max(
        tf.reduce_min(dist_mat, axis=1), axis=1)

    score = KeypointDiscriminator().build_model(
        tf.expand_dims(point_cloud, 2), keypoints_vae,
        funct_vect_vae)
    inference_graph = {'point_cloud_tf': point_cloud_tf,
                       'keypoints': keypoints_vae,
                       'funct_vect': funct_vect_vae,
                       'score': score,
                       'dist': dist}
    return inference_graph


def forward_grasp(point_cloud_tf,
                  grasp_keypoint,
                  num_points=1024,
                  num_samples=128,
                  dist_thres=0.3,
                  dist_kp_thres=0.4):
    """A forward pass that predicts the best grasp from
       the visual observation.
       
    Args:
        point_cloud_tf: A tensor of point cloud.
        grasp_keypoint: The grasp point. The grasp will be
            chosen near the grasp point.
        num_points: The number of points in the point cloud.
        num_samples: The number of grasp candidates produced
            by the grasp generator (VAE).
        dist_thres: The maximum allowed Chamfer distance 
            between the point cloud and the predicted grasp.
        dist_kp_thres: The maximum allowed distance between 
            the grasp point and the predicted the grasp.
        
    Returns:
        top_action: The predicted grasp with the highest score.
        top_score: The score of the predicted grasp.
    """
    point_cloud_tf = tf.reshape(
        point_cloud_tf, [1, num_points, 3])
    point_cloud = tf.tile(
        point_cloud_tf, [num_samples, 1, 1])
    latent_var = tf.concat(
        [tf.zeros([num_samples, 2], dtype=tf.float32),
         tf.ones([num_samples, 2], dtype=tf.float32)],
        axis=1)
    pose_vae_out = GraspDecoder().build_model(
        tf.reshape(point_cloud,
                   (-1, num_points, 1, 3)), latent_var)
    pose_rot = pose_vect_to_rot(pose_vae_out)
    pose_rot = tf.reshape(
        tf.transpose(pose_rot, [0, 2, 1]), (-1, 6))

    num_grasp = pose_rot.get_shape().as_list()[0]
    point_cloud_discr = tf.tile(
        point_cloud_tf, [num_grasp, 1, 1])

    pose_loc = tf.expand_dims(pose_rot[:, :3], 1)
    dist = tf.linalg.norm(
        tf.add(pose_loc[:, :, :2],
               - point_cloud_discr[:, :, :2]), axis=2)

    dist_min = tf.reduce_min(dist, axis=1)
    dist_mask = tf.less(dist_min, dist_thres)

    pose_loc = tf.squeeze(pose_loc, 1)
    dist_kp = tf.linalg.norm(
        pose_loc[:, :2] - grasp_keypoint[:, :2], axis=1)

    dist_kp_min = tf.reduce_min(dist_kp)
    dist_kp_mask = tf.less(dist_kp, dist_kp_thres)

    mask = tf.logical_or(
        tf.logical_and(dist_mask, dist_kp_mask),
        tf.equal(dist_kp_min, dist_kp))

    mask = tf.reshape(mask, [num_grasp])

    pose_rot = tf.boolean_mask(pose_rot, mask, axis=0)

    point_cloud_discr = tf.boolean_mask(
        point_cloud_discr, mask, axis=0)

    score = GraspDiscriminator().build_model(
        tf.expand_dims(point_cloud_discr, 2), pose_rot)

    index = tf.argmax(tf.reshape(score, [-1]), 0)
    top_score = score[index]
    top_action = pose_rot[index]

    return top_action, top_score


def forward_keypoint(point_cloud_tf,
                     num_points=1024,
                     num_samples=256,
                     dist_thres=0.4,
                     num_funct_vect=1,
                     funct_on_hull=True):
    """A forward pass that predicts the keypoints from 
       the visual observation.
       
    Args:
        point_cloud_tf: A tensor of point cloud.
        num_points: The number of points in the point cloud.
        num_samples: The number of keypoint candidates 
            produced by the keypoint generator.
        dist_thres: The maximum allowed Chamfer distance 
            between the keypoints and the point cloud.
        num_funct_vect: The number of vectors pointing from
            the function point to the effect point.
        funct_on_hull: Whether the function point should on
            the convex hull of the cluster centers of the 
            point cloud.
            
    Returns:
        top_keypoints: The predicted grasp point and function point.
        top_funct_vect: The predicted vector pointing from the function point to
            the effect point.
        top_score: The score of the keypoints.
    """

    point_cloud_tf = tf.reshape(
        point_cloud_tf, [1, num_points, 3])
    point_cloud = tf.tile(
        point_cloud_tf, [num_samples, 1, 1])
    latent_var = tf.concat(
        [tf.zeros([num_samples, 2], dtype=tf.float32),
         tf.ones([num_samples, 2], dtype=tf.float32)],
        axis=1)
    [keypoints_vae, funct_vect_vae] = KeypointDecoder(
    ).build_model(
        tf.reshape(point_cloud, (-1, num_points, 1, 3)),
        latent_var,
        num_funct_vect,
        truncated_normal=True)

    if num_funct_vect:
        v_mask = tf.constant([[[1, 1, 0]]], dtype=tf.float32)
        funct_vect_vae = funct_vect_vae * v_mask
        v_norm = tf.linalg.norm(funct_vect_vae, axis=-1, keepdims=True)
        funct_vect_vae = funct_vect_vae / (1e-6 + v_norm)

    dist_mat = tf.linalg.norm(
        tf.add(tf.expand_dims(point_cloud, 3),
               -tf.transpose(
            tf.concat([tf.expand_dims(k, 3) for
                       k in keypoints_vae], axis=1),
            [0, 3, 2, 1])), axis=2)
    dist = tf.reduce_max(
        tf.reduce_min(dist_mat, axis=1), axis=1)
    dist_min = tf.reduce_min(dist)

    mask = tf.logical_or(
        tf.less(dist, dist_thres),
        tf.equal(dist, dist_min))

    keypoints_vae = [tf.boolean_mask(k, mask, axis=0) for k in keypoints_vae]

    point_cloud_discr = tf.boolean_mask(point_cloud, mask, axis=0)

    if num_funct_vect:
        funct_vect_vae = tf.boolean_mask(funct_vect_vae, mask, axis=0)

    score = KeypointDiscriminator().build_model(
        tf.expand_dims(point_cloud_discr, 2), keypoints_vae, funct_vect_vae)

    index = tf.argmax(tf.reshape(score, [-1]), 0)
    top_score = score[index]
    top_keypoints = [k[index] for k in keypoints_vae]

    top_funct_vect = None
    if num_funct_vect:
        top_funct_vect = funct_vect_vae[index]

    grasp_point, funct_point = top_keypoints

    [grasp_point, funct_point
     ] = tf.py_func(rectify_keypoints,
                    [point_cloud_tf, grasp_point, funct_point, funct_on_hull],
                    [tf.float32, tf.float32])
    top_keypoints = [grasp_point, funct_point]

    return top_keypoints, top_funct_vect, top_score


def train_vae_grasp(data_path,
                    steps=60000,
                    batch_size=256,
                    eval_size=64,
                    l2_weight=1e-6,
                    log_step=20,
                    eval_step=6000,
                    save_step=6000,
                    model_path=None,
                    optimizer='Adam'):
    """Trains the VAE for grasp generation.
    
    Args:
        data_path: The training data in a single h5df file.
        steps: The total number of training steps.
        batch_size: The training batch size.
        eval_size: The evaluation batch size.
        l2_weight: The L2 regularization weight.
        log_step: The interval for logging.
        eval_step: The interval for evaluation.
        save_step: The interval for saving the model weights.
        model_path: The pretrained model.
        optimizer: Adam or SGDM.

    Returns:
        None.
    """
    loader = GraspReader(data_path)
    graph = build_grasp_training_graph()
    pose_rot_train = graph['pose_rot']
    graph_inf = build_grasp_inference_graph(num_samples=eval_size)
    point_cloud_tf_inf = graph_inf['point_cloud_tf']
    pose_rot_inf = graph_inf['pose_rot']

    learning_rate = tf.placeholder(tf.float32, shape=())
    point_cloud_tf = graph['point_cloud_tf']
    pose_tf = graph['pose_tf']
    loss_vae_loc = graph['loss_vae_loc']
    loss_vae_rot = graph['loss_vae_rot']
    loss_vae_mmd = graph['loss_vae_mmd'] * 0.007
    loss_vae_dist = graph['loss_vae_dist'] * 0.1
    z_mean = graph['z_mean']
    z_std = graph['z_std']

    std_gt_loc = graph['std_gt_loc']
    std_gt_rot = graph['std_gt_rot']

    weight_loss = [tf.nn.l2_loss(var) for var
                   in tf.trainable_variables()]
    weight_loss = tf.reduce_sum(weight_loss) * l2_weight

    loss_vae = loss_vae_loc + loss_vae_rot + \
        loss_vae_mmd + loss_vae_dist
    loss = weight_loss + loss_vae

    if optimizer == 'Adam':
        train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss)
    elif optimizer == 'SGDM':
        train_op = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9).minimize(loss)
    else:
        raise NotImplementedError

    all_vars = tf.get_collection_ref(
        tf.GraphKeys.GLOBAL_VARIABLES)
    var_list_vae = [var for var in all_vars
                    if 'vae_grasp' in var.name and
                       'Momentum' not in var.name and
                       'Adam' not in var.name]

    saver = tf.train.Saver(var_list=var_list_vae)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer()])
        if model_path:
            running_log.write('vae',
                              'loading model from {}'.format(model_path))
            saver.restore(sess, model_path)
        for step in range(steps + 1):
            pos_p_np, pos_g_np = loader.sample_pos_train(batch_size)
            pos_p_np, pos_g_np = loader.random_rotate(pos_p_np, pos_g_np)
            [_, loss_np, vae_loc,
             vae_rot, vae_mmd, vae_dist, weight, pose_rot_train_np,
             std_gt_loc_np, std_gt_rot_np,
             z_mean_np, z_std_np] = sess.run([
                 train_op, loss, loss_vae_loc,
                 loss_vae_rot, loss_vae_mmd, loss_vae_dist,
                 weight_loss, pose_rot_train, std_gt_loc,
                 std_gt_rot, z_mean, z_std],
                feed_dict={point_cloud_tf: pos_p_np,
                           pose_tf: pos_g_np,
                           learning_rate: get_learning_rate(step, steps)})
            if step % log_step == 0:
                running_log.write('vae',
                                  'step: {}/{}, '.format(step, steps) +
                                  'loss: {:.3f}, loc: {:.3f}/{:.3f}, '.format(
                                      loss_np, vae_loc, std_gt_loc_np) +
                                  'rot: {:.3f}/{:.3f}, '.format(
                                      vae_rot, std_gt_rot_np) +
                                  'mmd: {:.3f} ({:.3f} {:.3f}), '.format(
                                      vae_mmd, z_mean_np, z_std_np) +
                                  'dist: {:.3f}'.format(vae_dist))
            if step > 0 and step % save_step == 0:
                makedir('./runs/vae')
                saver.save(sess,
                           './runs/vae/vae_{}'.format(str(step).zfill(6)))
                for index in range(pos_p_np.shape[0]):
                    feed_dict = {point_cloud_tf_inf:
                                 pos_p_np[np.newaxis, index]}
                    pose_pred = sess.run([pose_rot_inf],
                                         feed_dict=feed_dict)[0]
                    visualize(pos_p_np[index],
                              pose_pred, './runs/vae/plot',
                              str(step).zfill(6) + '_' + str(index).zfill(3))

            if step > 0 and step % eval_step == 0:
                recall = []
                pos_p_np, pos_g_np = loader.sample_pos_val(
                    size=eval_size)
                for index in range(pos_p_np.shape[0]):
                    feed_dict = {point_cloud_tf_inf:
                                 pos_p_np[np.newaxis, index]}
                    pose_pred = sess.run([pose_rot_inf],
                                         feed_dict=feed_dict)[0]
                    recall.append(success_recall(
                        pose_pred, pos_g_np[np.newaxis, index]))
                running_log.write('vae',
                                  'Recall on validation set: {:.2f}'.format(
                                      np.mean(recall) * 100))

                recall = []
                pos_p_np, pos_g_np = loader.sample_pos_train(
                    size=eval_size)
                for index in range(pos_p_np.shape[0]):
                    feed_dict = {point_cloud_tf_inf:
                                 pos_p_np[np.newaxis, index]}
                    pose_pred = sess.run([pose_rot_inf],
                                         feed_dict=feed_dict)[0]
                    recall.append(success_recall(
                        pose_pred, pos_g_np[np.newaxis, index]))
                running_log.write('vae',
                                  'Recall on training set: {:.2f}'.format(
                                      np.mean(recall) * 100))


def train_vae_keypoint(data_path,
                       steps=120000,
                       batch_size=256,
                       eval_size=128,
                       l2_weight=1e-6,
                       log_step=20,
                       eval_step=4000,
                       save_step=4000,
                       model_path=None,
                       task_name='task',
                       optimizer='Adam'):
    """Trains the VAE for keypoint generation.
    
    Args:
        data_path: The training data in a single h5df file.
        steps: The total number of training steps.
        batch_size: The training batch size.
        eval_size: The evaluation batch size.
        l2_weight: The L2 regularization weight.
        log_step: The interval for logging.
        eval_step: The interval for evaluation.
        save_step: The interval for saving the model weights.
        model_path: The pretrained model.
        task_name: The name of the task to be trained on. This
            is only for distinguishing the name of log files and
            the saved model.
        optimizer: Adam or SGDM.

    Returns:
        None.
    """
    loader = KeypointReader(data_path)
    num_funct_vect = loader.num_funct_vect

    graph = build_keypoint_training_graph(num_funct_vect=num_funct_vect)
    learning_rate = tf.placeholder(tf.float32, shape=())

    point_cloud_tf = graph['point_cloud_tf']
    grasp_point_tf = graph['grasp_point_tf']
    funct_point_tf = graph['funct_point_tf']
    funct_vect_tf = graph['funct_vect_tf']

    loss_vae_grasp = graph['loss_vae_grasp']
    loss_vae_funct = graph['loss_vae_funct']
    loss_vae_funct_vect = graph['loss_vae_funct_vect']
    loss_vae_mmd = graph['loss_vae_mmd'] * 0.005

    z_mean = graph['z_mean']
    z_std = graph['z_std']

    std_gt_grasp = graph['std_gt_grasp']
    std_gt_funct = graph['std_gt_funct']
    std_gt_funct_vect = graph['std_gt_funct_vect']

    weight_loss = [tf.nn.l2_loss(var) for var
                   in tf.trainable_variables()]
    weight_loss = tf.reduce_sum(weight_loss) * l2_weight

    loss_vae = (loss_vae_grasp + loss_vae_funct +
                loss_vae_funct_vect + loss_vae_mmd)
    loss = weight_loss + loss_vae

    if optimizer == 'Adam':
        train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss)
    elif optimizer == 'SGDM':
        train_op = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9).minimize(loss)
    else:
        raise NotImplementedError

    all_vars = tf.get_collection_ref(
        tf.GraphKeys.GLOBAL_VARIABLES)
    var_list_vae = [var for var in all_vars
                    if 'vae_keypoint' in var.name and
                       'Momentum' not in var.name and
                       'Adam' not in var.name]

    saver = tf.train.Saver(var_list=var_list_vae)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer()])
        if model_path:
            running_log.write('vae',
                              'loading model from {}'.format(model_path))
            saver.restore(sess, model_path)

        for step in range(steps + 1):
            pos_p_np, pos_k_np = loader.sample_pos_train(batch_size)
            pos_p_np, pos_k_np = loader.random_rotate(pos_p_np, pos_k_np)

            pos_grasp_np, pos_funct_np, pos_funct_vect_np = np.split(
                pos_k_np, [1, 2], axis=1)
            feed_dict = {point_cloud_tf: pos_p_np,
                         grasp_point_tf: pos_grasp_np,
                         funct_point_tf: pos_funct_np,
                         learning_rate: get_learning_rate(step, steps)}
            if num_funct_vect:
                feed_dict[funct_vect_tf] = pos_funct_vect_np

            [_, loss_np, vae_grasp,
             vae_funct, vae_funct_vect, vae_mmd, weight,
             std_gt_grasp_np, std_gt_funct_np, std_gt_funct_vect_np,
             z_mean_np, z_std_np] = sess.run([
                 train_op, loss, loss_vae_grasp,
                 loss_vae_funct, loss_vae_funct_vect, loss_vae_mmd,
                 weight_loss, std_gt_grasp, std_gt_funct, std_gt_funct_vect,
                 z_mean, z_std],
                feed_dict=feed_dict)

            if step % log_step == 0:
                running_log.write('vae_{}'.format(task_name),
                                  'step: {}/{}, '.format(step, steps) +
                                  'loss: {:.3f}, grasp: {:.3f}/{:.3f}, '.format(
                                      loss_np, vae_grasp, std_gt_grasp_np) +
                                  'funct: {:.3f}/{:.3f}, '.format(
                                      vae_funct, std_gt_funct_np) +
                                  'vect: {:.3f}/{:.3f}, '.format(
                                      vae_funct_vect, std_gt_funct_vect_np) +
                                  'mmd: {:.3f} ({:.3f} {:.3f}), '.format(
                                      vae_mmd, z_mean_np, z_std_np))

            if step > 0 and step % save_step == 0:
                makedir('./runs/vae')
                saver.save(sess,
                           './runs/vae/vae_keypoint_{}_{}'.format(
                               task_name, str(step).zfill(6)))


def train_gcnn_grasp(data_path,
                     steps=60000,
                     batch_size=256,
                     eval_size=64,
                     l2_weight=1e-6,
                     log_step=20,
                     eval_step=6000,
                     save_step=6000,
                     model_path=None,
                     optimizer='SGDM',
                     lr_init=8e-4):
    """Trains the grasp evaluation network.
    
    Args:
        data_path: The training data in a single h5df file.
        steps: The total number of training steps.
        batch_size: The training batch size.
        eval_size: The evaluation batch size.
        l2_weight: The L2 regularization weight.
        log_step: The interval for logging.
        eval_step: The interval for evaluation.
        save_step: The interval for saving the model weights.
        model_path: The pretrained model.
        optimizer: Adam or SGDM.
        lr_init: The initial learning rate.

    Returns:
        None.
    """

    loader = GraspReader(data_path)
    graph = build_grasp_training_graph()
    learning_rate = tf.placeholder(tf.float32, shape=())
    point_cloud_tf = graph['point_cloud_tf']
    pose_tf = graph['pose_tf']
    pose_label_tf = graph['pose_label_tf']
    loss_discr_pos = graph['loss_discr_pos']
    acc_discr = graph['acc_discr']
    prec_discr = graph['prec_discr']
    aligned_point_cloud = graph['aligned_point_cloud']
    weight_loss = [tf.nn.l2_loss(var) for var
                   in tf.trainable_variables()]
    weight_loss = tf.reduce_sum(weight_loss) * l2_weight
    loss_discr = loss_discr_pos
    loss = weight_loss + loss_discr

    if optimizer == 'Adam':
        train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss)
    elif optimizer == 'SGDM':
        train_op = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9).minimize(loss)
    else:
        raise NotImplementedError

    all_vars = tf.get_collection_ref(
        tf.GraphKeys.GLOBAL_VARIABLES)
    var_list_gcnn = [var for var in all_vars
                     if 'grasp_discriminator' in var.name and
                        'Momentum' not in var.name and
                        'Adam' not in var.name]
    saver = tf.train.Saver(var_list=var_list_gcnn)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer()])
        if model_path:
            running_log.write('gcnn',
                              'loading model from {}'.format(model_path))
            saver.restore(sess, model_path)
        for step in range(steps + 1):
            pos_p_np, pos_g_np = loader.sample_pos_train(batch_size // 2)
            neg_p_np, neg_g_np = loader.sample_neg_train(batch_size // 2)
            p_np = np.concatenate([pos_p_np, neg_p_np], axis=0)
            g_np = np.concatenate([pos_g_np, neg_g_np], axis=0)
            pose_label_np = np.concatenate([np.ones(pos_g_np.shape[0]),
                                            np.zeros(neg_g_np.shape[0])], axis=0
                                           ).reshape([-1, 1]).astype(np.float32)
            [_, loss_np, acc_np] = sess.run([
                train_op, loss, acc_discr],
                feed_dict={point_cloud_tf: p_np,
                           pose_tf: g_np,
                           pose_label_tf: pose_label_np,
                           learning_rate: get_learning_rate(
                               step, steps, init=lr_init)})
            if step % log_step == 0:
                running_log.write('gcnn',
                                  'step: {}/{}, '.format(step, steps) +
                                  'loss: {:.3f}, acc: {:.3f}'.format(loss_np,
                                                                     acc_np))

            if step > 0 and step % save_step == 0:
                makedir('./runs/gcnn')
                saver.save(sess, './runs/gcnn/gcnn_{}'.format(
                    str(step).zfill(6)))

            if step > 0 and step % eval_step == 0:
                pos_p_np, pos_g_np = loader.sample_pos_val(eval_size // 2)
                neg_p_np, neg_g_np = loader.sample_neg_val(eval_size // 2)
                p_np = np.concatenate([pos_p_np, neg_p_np], axis=0)
                g_np = np.concatenate([pos_g_np, neg_g_np], axis=0)
                pose_label_np = np.concatenate(
                    [np.ones(pos_g_np.shape[0]), np.zeros(
                        neg_g_np.shape[0])], axis=0
                ).reshape([-1, 1]).astype(np.float32)
                [acc_np, prec_np, aligned_p_np] = sess.run(
                    [acc_discr, prec_discr, aligned_point_cloud],
                    feed_dict={point_cloud_tf: p_np,
                               pose_tf: g_np,
                               pose_label_tf: pose_label_np})
                running_log.write(
                    'gcnn', 'Validation acc: {:.2f}, prec: {:.2f}'.format(
                        acc_np * 100, prec_np * 100))

                for index in range(pos_p_np.shape[0]):
                    visualize(pos_p_np[index],
                              pos_g_np[np.newaxis, index],
                              './runs/gcnn/plot',
                              str(step).zfill(6) + '_' + str(index).zfill(3))
                    visualize(aligned_p_np[index],
                              np.array([[0, 0, 0, 0, 0, 0]]),
                              './runs/gcnn/plot',
                              str(step).zfill(6) + '_' +
                              str(index).zfill(3) + '_aligned')


def load_samples(loader, batch_size, stage, noise_level=0.2):
    """Load training and evaluation data.
    
    Args:
        loader: The Reader instance.
        batch_size: The batch size for training or evaluation.
        stage: 'train' or 'val'.
        noise_level: The noise to be added to the negative examples.

    Returns:
        p_np: A numpy array of point cloud.
        grasp_np: A numpy array of grasp point.
        funct_np: A numpy array of function point.
        funct_vect_np: A numpy array of the vector pointing from the 
            function point to the effect point.
        label: The binary success label of the keypoints given the 
            point cloud as visual observation.
    """
    if stage == 'train':
        pos_p_np, pos_k_np = loader.sample_pos_train(batch_size // 2)
        neg_p_np, neg_k_np = loader.sample_neg_train(batch_size // 2)
    elif stage == 'val':
        pos_p_np, pos_k_np = loader.sample_pos_val(batch_size // 2)
        neg_p_np, neg_k_np = loader.sample_neg_val(batch_size // 2)
    else:
        raise NotImplementedError

    pos_p_np, pos_k_np = loader.random_rotate(pos_p_np, pos_k_np)
    num_pos, num_neg = pos_p_np.shape[0], neg_p_np.shape[0]
    label_np = np.concatenate(
        [np.ones(shape=(num_pos, 1)),
         np.zeros(shape=(num_neg, 1))],
        axis=0).astype(np.float32)
    p_np = np.concatenate([pos_p_np, neg_p_np], axis=0)
    k_np = np.concatenate([pos_k_np, neg_k_np], axis=0)
    grasp_np, funct_np, funct_vect_np = np.split(
        k_np, [1, 2], axis=1)
    return p_np, grasp_np, funct_np, funct_vect_np, label_np


def train_discr_keypoint(data_path,
                         steps=120000,
                         batch_size=128,
                         eval_size=128,
                         l2_weight=1e-6,
                         log_step=20,
                         eval_step=4000,
                         save_step=4000,
                         model_path=None,
                         task_name='task',
                         optimizer='Adam'):
    """Trains the keypoint evaluation network.
    
    Args:
        data_path: The training data in a single h5df file.
        steps: The total number of training steps.
        batch_size: The training batch size.
        eval_size: The evaluation batch size.
        l2_weight: The L2 regularization weight.
        log_step: The interval for logging.
        eval_step: The interval for evaluation.
        save_step: The interval for saving the model weights.
        model_path: The pretrained model.
        task_name: The name of the task to be trained on. This
            is only for distinguishing the name of log files and
            the saved model.
        optimizer: 'Adam' or 'SGDM'.

    Returns:
        None.
    """
    loader = KeypointReader(data_path)
    num_funct_vect = loader.num_funct_vect

    graph = build_keypoint_training_graph(
        num_funct_vect=num_funct_vect)
    learning_rate = tf.placeholder(tf.float32, shape=())

    point_cloud_tf = graph['point_cloud_tf']
    grasp_point_tf = graph['grasp_point_tf']
    funct_point_tf = graph['funct_point_tf']
    funct_vect_tf = graph['funct_vect_tf']

    keypoints_label_tf = graph['keypoints_label_tf']
    loss_discr = graph['loss_discr']
    acc_discr = graph['acc_discr']

    weight_loss = [tf.nn.l2_loss(var) for var
                   in tf.trainable_variables()]
    weight_loss = tf.reduce_sum(weight_loss) * l2_weight

    loss = weight_loss + loss_discr

    if optimizer == 'Adam':
        train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss)
    elif optimizer == 'SGDM':
        train_op = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9).minimize(loss)
    else:
        raise NotImplementedError

    all_vars = tf.get_collection_ref(
        tf.GraphKeys.GLOBAL_VARIABLES)
    var_list_vae = [var for var in all_vars
                    if 'keypoint_discriminator' in var.name and
                       'Momentum' not in var.name and
                       'Adam' not in var.name]

    saver = tf.train.Saver(var_list=var_list_vae)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer()])
        if model_path:
            running_log.write('discr_{}'.format(task_name),
                              'loading model from {}'.format(model_path))
            saver.restore(sess, model_path)

        for step in range(steps + 1):
            p_np, grasp_np, funct_np, funct_vect_np, label_np = load_samples(
                loader, batch_size, 'train', 0.2)
            feed_dict = {point_cloud_tf: p_np,
                         grasp_point_tf: grasp_np,
                         funct_point_tf: funct_np,
                         keypoints_label_tf: label_np,
                         learning_rate: get_learning_rate(step, steps)}
            if num_funct_vect:
                feed_dict.update({funct_vect_tf: funct_vect_np})

            [_, loss_np, acc_np, weight
             ] = sess.run([
                 train_op, loss, acc_discr, weight_loss],
                feed_dict=feed_dict)

            if step % log_step == 0:
                running_log.write('discr_{}'.format(task_name),
                                  'step: {}/{}, '.format(step, steps) +
                                  'loss: {:.3f}, acc: {:.3f}'.format(
                                      loss_np, acc_np * 100))

            if step > 0 and step % save_step == 0:
                makedir('./runs/discr')
                saver.save(sess,
                           './runs/discr/discr_keypoint_{}_{}'.format(
                               task_name, str(step).zfill(6)))

            if step > 0 and step % eval_step == 0:
                for noise_level in [0.1, 0.2, 0.4, 0.8]:
                    [p_np, grasp_np, funct_np,
                        funct_vect_np, label_np] = load_samples(
                        loader, batch_size, 'train', noise_level)
                    feed_dict = {point_cloud_tf: p_np,
                                 grasp_point_tf: grasp_np,
                                 funct_point_tf: funct_np,
                                 keypoints_label_tf: label_np}
                    if num_funct_vect:
                        feed_dict.update({funct_vect_tf: funct_vect_np})

                    [acc_np] = sess.run([acc_discr], feed_dict=feed_dict)
                    running_log.write('discr_{}'.format(task_name),
                                      'noise: {:.3f}, acc: {:.3f}'.format(
                                          noise_level, acc_np * 100))


def inference_grasp(data_path,
                    model_path,
                    batch_size=128,
                    score_thres=0.7,
                    dist_thres=0.2,
                    show_best=True):
    """Predicts the grasps offline.
    
    Args:
        data_path: The point cloud data in hdf5.
        model_path: The pre-trained model.
        batch_size: The batch size in inference.
        score_thres: The minimum allowed score of the predicted grasps.
        dist_thres: The maximum allowed Chamfer distance between the
            grasp and the input point cloud.
        show_best: Whether to only visualize the best predicted grasp.

    Returns:
        None.
    """
    loader = GraspReader(data_path)
    graph = build_grasp_inference_graph(num_samples=batch_size)
    point_cloud_tf = graph['point_cloud_tf']
    pose_rot = graph['pose_rot']
    score = graph['score']
    dist_min = graph['dist_min']

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        pos_p_np, pos_g_np = loader.sample_pos_val(
            size=batch_size)
        batch_size = pos_p_np.shape[0]

        for index in range(batch_size):
            running_log.write(
                'vae_gcnn', 'Inference {} / {}'.format(index + 1, batch_size))
            feed_dict = {point_cloud_tf:
                         pos_p_np[np.newaxis, index]}
            pose_pred, score_pred, dist_pred = sess.run(
                [pose_rot, score, dist_min],
                feed_dict=feed_dict)
            score_pred = np.squeeze(score_pred)
            keep = np.logical_and(score_pred > score_thres,
                                  dist_pred < dist_thres)
            pose_pred = pose_pred[keep]
            score_pred = score_pred[keep]

            if show_best:
                indices = np.argsort(score_pred)[::-1]
                pose_pred = pose_pred[np.newaxis, indices[0]]

            visualize(pos_p_np[index], pose_pred,
                      './runs/vae_gcnn/plot',
                      str(index).zfill(6))

            visualize(pos_p_np[index],
                      pos_g_np[np.newaxis, index],
                      './runs/vae_gcnn/plot',
                      str(index).zfill(6) + '_ref')

    return


def inference_keypoint(data_path,
                       model_path,
                       batch_size=128,
                       num_points=1024):
    """Predicts the keypoints offline.
    
    Args:
        data_path: The point cloud data in hdf5.
        model_path: The pre-trained model.
        batch_size: The batch size in inference.
        num_points: The points in a single frame of point cloud.

    Returns:
        None.
    """
    loader = KeypointReader(data_path)
    point_cloud_tf = tf.placeholder(
        dtype=tf.float32, shape=[1, num_points, 3])
    keypoints, score = forward_keypoint(point_cloud_tf)
    grasp_point, funct_point = keypoints
    [grasp_point, funct_point
     ] = tf.py_func(rectify_keypoints,
                    [point_cloud_tf, grasp_point, funct_point],
                    [tf.float32, tf.float32])
    keypoints = [grasp_point, funct_point]

    score = tf.nn.sigmoid(score)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        pos_p_np, pos_k_np = loader.sample_pos_train(
            size=batch_size)
        batch_size = pos_p_np.shape[0]
        scores = []
        for index in range(batch_size):
            running_log.write(
                'keypoint', 'Inference {} / {}'.format(index + 1, batch_size))
            feed_dict = {point_cloud_tf:
                         pos_p_np[np.newaxis, index]}
            keypoints_np, score_np = sess.run(
                [keypoints, score], feed_dict=feed_dict)
            visualize_keypoints(pos_p_np[index],
                                keypoints_np,
                                './runs/keypoint',
                                str(index).zfill(6))
            scores.append(score_np)
        running_log.write('keypoint', 'Mean score: {}'.format(np.mean(scores)))
    return
