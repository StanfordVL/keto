#!/usr/bin/env python

"""Normalize a .obj mesh.
"""

import argparse
import glob
import os
import time

import numpy as np

from utils import mesh_utils


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input',
            dest='input_pattern',
            type=str,
            required=True,
            help='The input path pattern.')

    parser.add_argument(
            '--output',
            dest='output_dir',
            type=str,
            required=True,
            help='The output directory')

    args = parser.parse_args()

    return args


def normalize_mesh(input_path, output_path):
    """
    """
    # Load.
    vertices, faces = mesh_utils.read_from_obj(input_path)
    vertices = np.array(vertices)
    faces = np.array(faces)
    centroid = mesh_utils.compute_centroid(vertices, faces)
    span = vertices.max(axis=0) - vertices.min(axis=0)

    # Shift the centroid to the origin.
    vertices -= np.array([centroid])

    # Align the longest dimension with z-axis.
    axis_order = span.argsort()
    vertices = vertices[:, axis_order]
    faces = faces[:, axis_order]
    span = span[axis_order]

    # Scale to fit a unit cube.
    vertices /= np.array([span])

    # Save to the output directory.
    mesh_utils.write_to_obj(output_path, vertices, faces)


def main():
    args = parse_args()

    input_paths = glob.glob(args.input_pattern)

    if not os.path.exists(args.output_dir):
        print('Creating output directory %s...' % (args.output_dir))
        os.makedirs(args.output_dir)

    tic = time.time()

    for index, input_path in enumerate(input_paths):
        # basename = os.path.splitext(os.path.basename(input_path))[0]
        # output_path = os.path.join(args.output_dir, '%s.obj' % basename)
        print('Processing file %s...' % (input_path))
        output_path = os.path.join(args.output_dir, '%06d.obj' % index)
        normalize_mesh(input_path, output_path)

    toc = time.time()
    print('Finished in %.2f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
