"""Visualization utilities for grasping.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import matplotlib.pyplot as plt
except Exception:
    print('Warning: Failed to import matplotlib.pyplot')

import cv2
import numpy as np


def plot_grasp(
        grasp,
        color='r',
        arrow_len=4,
        arrow_head_len=2,
        arrow_head_width=3,
        arrow_width=1,
        jaw_len=3,
        jaw_width=3.0,
        grasp_center_size=7.5,
        grasp_center_thickness=2.5,
        grasp_center_style='+',
        grasp_axis_width=1,
        grasp_axis_style='--',
        line_width=8.0,
        show_center=True,
        show_axis=False,
        scale=1.0):
    """Plots a 2D grasp with arrow and jaw.
    
    Args:
    grasp: 2D grasp to plot.
    color: Color of plotted grasp.
    arrow_len: Length of arrow body.
    arrow_head_len: Length of arrow head.
    arrow_head_width: Width of arrow head.
    arrow_width: Width of arrow body.
    jaw_len: Length of jaw line.
    jaw_width: Line width of jaw line.
    grasp_center_thickness: Thickness of grasp center.
    grasp_center_style: Style of center of grasp.
    grasp_axis_width: Line width of grasp axis.
    grasp_axis_style: Style of grasp axis line.
    show_center: Whether or not to plot the grasp center.
    show_axis: Whether or not to plot the grasp axis.
    """
    # Plot grasp center
    if show_center:
        plt.plot(grasp.center[0],
                 grasp.center[1],
                 c=color,
                 marker=grasp_center_style,
                 mew=scale * grasp_center_thickness,
                 ms=scale * grasp_center_size)
    
    # Compute axis and jaw locations.
    axis = grasp.axis
    g1 = grasp.center - 0.5 * float(grasp.width_pixel) * axis
    g2 = grasp.center + 0.5 * float(grasp.width_pixel) * axis
    # Start location of grasp jaw 1.
    g1p = g1 - scale * arrow_len * axis  
    # Start location of grasp jaw 2.
    g2p = g2 + scale * arrow_len * axis  

    # Plot grasp axis.
    if show_axis:
        plt.plot([g1[0], g2[0]], [g1[1], g2[1]],
                 color=color,
                 linewidth=scale * grasp_axis_width,
                 linestyle=grasp_axis_style)
    
    # Direction of jaw line.
    jaw_dir = scale * jaw_len * np.array([axis[1], -axis[0]])
    
    # Length of arrow.
    alpha = scale * (arrow_len - arrow_head_len)
    
    # Plot jaw 1.
    jaw_line1 = np.c_[g1 + jaw_dir, g1 - jaw_dir].T
    plt.plot(jaw_line1[:, 0],
             jaw_line1[:, 1],
             linewidth=scale * jaw_width,
             c=color) 
    plt.arrow(g1p[0],
              g1p[1],
              alpha * axis[0],
              alpha * axis[1],
              width=scale * arrow_width,
              head_width=scale * arrow_head_width,
              head_length=scale * arrow_head_len,
              fc=color,
              ec=color)

    # Plot jaw 2.
    jaw_line2 = np.c_[g2 + jaw_dir, g2 - jaw_dir].T
    plt.plot(jaw_line2[:, 0],
             jaw_line2[:, 1],
             linewidth=scale * jaw_width,
             c=color) 
    plt.arrow(g2p[0],
              g2p[1],
              -alpha * axis[0],
              -alpha * axis[1],
              width=scale * arrow_width,
              head_width=scale * arrow_head_width,
              head_length=scale * arrow_head_len,
              fc=color,
              ec=color)


def plot_grasp_on_image(image, grasp, grasp_quality=None):
    """Visualize the final grasp.

    Args:
        image: The whole image captured from the camera.
        grasp: The chosen Grasp2D instance.
        grasp_quality: The corresponding grasp quality.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=plt.cm.gray_r)
    plot_grasp(grasp, scale=1.5, show_center=False, show_axis=True)

    if grasp_quality:
        plt.title('Planned grasp on depth (Q=%f)' % (grasp_quality))

    plt.show() 


def visualize_sampled_grasps(grasp_images, grasp_poses, grasp_qualities,
                             num_sampled_grasps, task_qualities=None):
    """Visualize sampled grasps.

    Args:
        grasp_images: A list or array of grasp images.
        grasp_poses: A list of array of grasp poses.
        grasp_qualities: The predicted grasp qualities.
    """
    sorted_indices = np.argsort(grasp_qualities.flatten())[::-1]

    plt.figure(figsize=(10, 10))
    d = int(np.ceil(np.sqrt(num_sampled_grasps)))

    for i in range(min(num_sampled_grasps, len(sorted_indices))):
        ind = sorted_indices[i]
        plt.subplot(d, d, i + 1)
        plt.imshow(grasp_images[ind].squeeze(axis=-1),
                   cmap=plt.cm.gray_r)
        if task_qualities is None:
            # Visualize purposeless grasps.
            plt.title('id: %d\ndepth: %.2f\nquality: %.2f'
                      % (i + 1, grasp_poses[ind], grasp_qualities[ind]))
        else:
            # Visualize task-oriented grasps.
            plt.title('id: %d\ngrasp: %.2f\ntask: %.2f'
                      % (i + 1, grasp_qualities[ind], task_qualities[ind]))

    plt.tight_layout()
    plt.show() 


def visualize_heatmap(camera_image,
                      grasps,
                      grasp_qualities,
                      task_qualities=None,
                      kernel_size=11):
    """Visualize the final grasp.

    Args:
        camera_image: The whole image captured from the camera.
        grasps: The chosen Grasp2D instance.
        grasp_qualities: The corresponding grasp quality.
        task_qualities: The corresponding task quality.
    """
    plt.figure(figsize=(10, 20))

    plt.subplot(1, 2, 1)
    plt.imshow(camera_image, cmap=plt.cm.gray_r)

    plt.subplot(1, 2, 2)
    image_shape = camera_image.shape
    heatmap = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)

    for i, grasp in enumerate(grasps):
        center = grasp.center
        score = grasp_qualities[i]
        heatmap[int(center[1]), int(center[0])] += score
    
    heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), 0)
    heatmap /= np.sum(heatmap)
    plt.imshow(heatmap)

    plt.show() 
