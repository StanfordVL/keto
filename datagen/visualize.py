#!/usr/bin/env python

"""Visualize a URDF or OBJ file in the simulation.
"""

import argparse
import os

import pybullet


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input',
            dest='input_path',
            type=str,
            default=None,
            help='The path to the URDF file.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    _, file_extension = os.path.splitext(args.input_path)

    if file_extension == '.obj':
        with open('templates/simple.xml', 'r') as f:
            template = f.read()

        urdf_path = 'outputs/tmp.urdf'
        filename = os.path.relpath(args.input_path, urdf_path)
        filename = os.path.relpath(filename, '..')
        urdf = template.format(filename=filename)

        with open(urdf_path, 'w') as f:
            f.write(urdf)
            f.flush()
    else:
        urdf_path = args.input_path

    pybullet.connect(pybullet.GUI)
    pybullet.resetSimulation()
    pybullet.setRealTimeSimulation(0)
    pybullet.setTimeStep(1e-2)

    pybullet.loadURDF(
            fileName=urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            )

    while(1):
        pybullet.stepSimulation()


if __name__ == '__main__':
    main()
