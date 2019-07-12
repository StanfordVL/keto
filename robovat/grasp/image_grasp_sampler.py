"""Image grasp samplers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.misc
import scipy.spatial.distance
import scipy.ndimage.filters
from PIL import Image

from robovat.perception import depth_utils
from robovat.perception.camera.camera import Camera
from robovat.utils.logging import logger
from robovat.grasp import visualize as grasp_vis
from robovat.grasp import Grasp2D


def surface_normals(depth, edge_pixels):
    """Return an array of the surface normals at the edge pixels.
    """
    # Compute the gradients.
    grad = np.gradient(depth.astype(np.float32))

    # Compute surface normals.
    normals = np.zeros([edge_pixels.shape[0], 2])

    for i, pixel in enumerate(edge_pixels):
        dx = grad[1][pixel[0], pixel[1]]
        dy = grad[0][pixel[0], pixel[1]]

        normal = np.array([dy, dx])

        if np.linalg.norm(normal) == 0:
            normal = np.array([1, 0])

        normal = normal / np.linalg.norm(normal)
        normals[i, :] = normal

    return normals


def force_closure(p1, p2, n1, n2, mu):
    """Check if the point and normal pairs are in force closure.
    """
    # Line between the contacts.
    v = p2 - p1
    v = v / np.linalg.norm(v)
    
    # Compute cone membership.
    alpha = np.arctan(mu)
    in_cone_1 = (np.arccos(n1.dot(-v)) < alpha)
    in_cone_2 = (np.arccos(n2.dot(v)) < alpha)

    return (in_cone_1 and in_cone_2)


class ImageGraspSampler(object):
    """Image grasp sampler.

    Wraps image to crane grasp candidate generation for easy deployment of
    GQ-CNN.
    """

    __metaclass__ = ABCMeta

    def sample(self, depth, camera, num_samples):
        """Samples a set of 2D grasps from a given RGB-D image.
        
        Args:
            depth: Depth image.
            camera: The camera model.
            num_samples: Number of grasps to sample.
 
        Returns:
            List of 2D grasp candidates
        """
        # Sample an initial set of grasps (without depth).
        logger.debug('Sampling grasp candidates...')
        grasps = self._sample(depth, camera, num_samples)
        logger.debug('Sampled %d grasp candidates from the image.'
                     % (len(grasps)))

        return grasps

    @abstractmethod
    def _sample(self, depth, camera, num_samples):
        """Sample a set of 2D grasp candidates from a depth image.

        Args:
            depth: Depth image.
            camera: The camera model.
            num_samples: Number of grasps to sample.
 
        Returns:
            List of 2D grasp candidates
        """
        pass
        

class AntipodalDepthImageGraspSampler(ImageGraspSampler):
    """Grasp sampler for antipodal point pairs from depth image gradients.
    """

    def __init__(self, 
                 friction_coef,
                 depth_grad_thresh,
                 depth_grad_gaussian_sigma,
                 downsample_rate,
                 max_rejection_samples,
                 boundary,
                 min_dist_from_boundary,
                 min_grasp_dist,
                 angle_dist_weight,
                 depth_samples_per_grasp,
                 min_depth_offset,
                 max_depth_offset,
                 depth_sample_window_height,
                 depth_sample_window_width,
                 gripper_width=0.0,
                 debug=False):
        """Initialize the sampler. 

        Args:
            friction_coef: Friction coefficient for 2D force closure.
            depth_grad_thresh: Threshold for depth image gradients to determine
                edge points for sampling.
            depth_grad_gaussian_sigma: Sigma used for pre-smoothing the depth
            image for better gradients.
            downsample_rate: Factor to downsample the depth image by before
                sampling grasps.
            max_rejection_samples: Ceiling on the number of grasps to check in
                antipodal grasp rejection sampling.
            boundary: The rectangular boundary of the grasping region on images.
            min_dist_from_boundary: Minimum distance from the boundary of the
                grasping region
            min_grasp_dist: Threshold on the grasp distance.
            angle_dist_weight: Amount to weight the angle difference in grasp
                distance computation.
            depth_samples_per_grasp: Number of depth samples to take per grasp.
            min_depth_offset: Offset from the minimum depth at the grasp center
                pixel to use in depth sampling.
            max_depth_offset: Offset from the maximum depth across all edges.
            depth_sample_window_height: Height of a window around the grasp
                center pixel used to determine min depth.
            depth_sample_window_width: Width of a window around the grasp center
                pixel used to determine min depth.
            gripper_width: Maximum width of the gripper.
            debug: If true, visualize the grasps for debugging.
        """
        # Antipodality parameters.
        self.friction_coef = friction_coef
        self.depth_grad_thresh = depth_grad_thresh
        self.depth_grad_gaussian_sigma = depth_grad_gaussian_sigma
        self.downsample_rate = downsample_rate
        self.max_rejection_samples = max_rejection_samples

        # Distance thresholds for rejection sampling.
        self.boundary = boundary
        self.min_dist_from_boundary = min_dist_from_boundary
        self.min_grasp_dist = min_grasp_dist
        self.angle_dist_weight = angle_dist_weight

        # Depth sampling parameters.
        self.depth_samples_per_grasp = max(depth_samples_per_grasp, 1)
        self.min_depth_offset = min_depth_offset
        self.max_depth_offset = max_depth_offset
        self.depth_sample_window_height = depth_sample_window_height
        self.depth_sample_window_width = depth_sample_window_width

        # Gripper width.
        self.gripper_width = gripper_width

        self.debug = debug

    def _sample(self, depth, camera, num_samples):
        """Sample antipodal grasps.

        Sample a set of 2D grasp candidates from a depth image by finding depth
        edges, then uniformly sampling point pairs and keeping only antipodal
        grasps with width less than the maximum allowable.

        Args:
            depth: Depth image.
            camera: The camera model.
            num_samples: Number of grasps to sample.
 
        Returns:
            List of 2D grasp candidates
        """
        if not isinstance(camera, Camera):
            intrinsics = camera
            camera = Camera()
            camera.set_calibration(intrinsics, np.zeros((3,)), np.zeros((3,)))

        # Crope the image.
        depth_cropped = depth[self.boundary[0]:self.boundary[1],
                              self.boundary[2]:self.boundary[3]]
        depth_filtered = scipy.ndimage.filters.gaussian_filter(
            depth_cropped, sigma=self.depth_grad_gaussian_sigma)

        # Compute edge pixels.
        depth_downsampled = np.array(
            Image.fromarray(depth_filtered).resize(
                (int(depth_filtered.shape[1] / self.downsample_rate),
                 int(depth_filtered.shape[0] / self.downsample_rate)),
                Image.BILINEAR))
        depth_threshed = depth_utils.threshold_gradients(
            depth_downsampled, self.depth_grad_thresh)
        depth_zero = np.where(depth_threshed == 0)
        depth_zero = np.c_[depth_zero[0], depth_zero[1]]
        edge_pixels = self.downsample_rate * depth_zero

        # Return if no edge pixels
        num_pixels = edge_pixels.shape[0]
        if num_pixels == 0:
            return []

        # Compute surface normals.
        edge_normals = surface_normals(depth_filtered, edge_pixels)

        # Prune surface normals. Form set of valid candidate point pairs.
        if self.gripper_width > 0:
            max_grasp_width_pixel = Grasp2D(
                np.zeros(2),
                0.0,
                depth=np.max(depth_filtered) + self.min_depth_offset,
                width=self.gripper_width,
                camera=camera,
                ).width_pixel
        else:
            max_grasp_width_pixel = np.inf

        normal_ip = edge_normals.dot(edge_normals.T)
        distances = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(edge_pixels))
        valid_indices = np.where(
            (normal_ip < -np.cos(np.arctan(self.friction_coef))) &
            (distances < max_grasp_width_pixel) &
            (distances > 0.0))
        valid_indices = np.c_[valid_indices[0], valid_indices[1]]

        # Return if no antipodal pairs.
        num_pairs = valid_indices.shape[0]
        if num_pairs == 0:
            return []

        # Iteratively sample grasps.
        grasps = []
        sample_size = min(self.max_rejection_samples, num_pairs)
        candidate_pair_indices = np.random.choice(
            num_pairs, size=sample_size, replace=False)

        for sample_ind in candidate_pair_indices: 

            if len(grasps) >= num_samples:
                break

            # Sample a random pair without replacement.
            pair_ind = valid_indices[sample_ind, :]
            p1 = edge_pixels[pair_ind[0], :]
            p2 = edge_pixels[pair_ind[1], :]
            n1 = edge_normals[pair_ind[0], :]
            n2 = edge_normals[pair_ind[1], :]

            # Check the force closure.
            if not force_closure(p1, p2, n1, n2, self.friction_coef):
                continue

            # Compute grasp parameters.
            grasp_center_pixel = (p1 + p2) / 2
            grasp_center_pixel[0] += self.boundary[0]
            grasp_center_pixel[1] += self.boundary[2]

            dist_from_boundary = min(
                np.abs(self.boundary[0] - grasp_center_pixel[0]),
                np.abs(self.boundary[2] - grasp_center_pixel[1]),
                np.abs(grasp_center_pixel[0] - self.boundary[1]),
                np.abs(grasp_center_pixel[1] - self.boundary[2]))

            if dist_from_boundary < self.min_dist_from_boundary:
                continue

            grasp_axis = p2 - p1
            grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)

            if grasp_axis[1] != 0:
                grasp_angle = np.arctan(grasp_axis[0] / grasp_axis[1])
            else:
                grasp_angle = 0
                
            # Form grasp object.
            grasp_center_point = np.array([grasp_center_pixel[1],
                                           grasp_center_pixel[0]])
            grasp = Grasp2D(center=grasp_center_point,
                            angle=grasp_angle,
                            depth=0.0,
                            width=self.gripper_width,
                            camera=camera)
            
            # Skip if the grasp is close to any previously sampled grasp.
            if len(grasps) > 0:
                grasp_dists = [
                    Grasp2D.image_dist(grasp,
                                       other_grasp,
                                       alpha=self.angle_dist_weight)
                    for other_grasp in grasps]

                if np.min(grasp_dists) <= self.min_grasp_dist:
                    continue

            # Get depth in the neighborhood of the center pixel.
            window = [
                int(grasp_center_pixel[0] - self.depth_sample_window_height),
                int(grasp_center_pixel[0] + self.depth_sample_window_height),
                int(grasp_center_pixel[1] - self.depth_sample_window_width),
                int(grasp_center_pixel[1] + self.depth_sample_window_width)]
            depth_window = depth[window[0]:window[1], window[2]:window[3]]
            center_depth = np.min(depth_window)

            if center_depth == 0 or np.isnan(center_depth):
                continue

            min_depth = np.min(center_depth) + self.min_depth_offset
            max_depth = np.max(center_depth) + self.max_depth_offset

            # Sample depth between the min and max.
            for i in range(self.depth_samples_per_grasp):
                sample_depth = (
                    min_depth + np.random.rand() * (max_depth - min_depth))
                grasp = Grasp2D(center=grasp_center_point,
                                angle=grasp_angle,
                                depth=sample_depth,
                                width=self.gripper_width,
                                camera=camera)
                grasps.append(grasp)

        # if self.debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(depth_cropped)
        plt.figure()
        plt.imshow(depth_downsampled)
        plt.figure()
        plt.imshow(depth_threshed)
        grasp_vis.plot_grasp_on_image(depth, grasps[0])
        plt.show()

        grasps = np.array(
            [g.vector for g in grasps], dtype=np.float32)

        return grasps
