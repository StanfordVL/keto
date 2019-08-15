#!/usr/bin/env python

"""Generate bodies.
"""

import argparse
import glob
import os
import time

import bodies


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--obj',
                        dest='obj_paths',
                        help='Path to the mesh files.',
                        default=None,
                        type=str)

    parser.add_argument('--color',
                        dest='color',
                        help='Color code for the object.',
                        default=None,
                        type=str)

    parser.add_argument('--output',
                        dest='output_dir',
                        help='The output directory.',
                        type=str,
                        required=True)

    parser.add_argument('--num',
                        dest='num_bodies',
                        help='The number of bodies to genearte.',
                        type=int,
                        required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    body_class = bodies.General

    if args.obj_paths is None:
        obj_paths = None
    else:
        obj_paths = glob.glob(args.obj_paths)

    body_generator = body_class(name='body',
                                obj_paths=obj_paths,
                                color=args.color)

    tic = time.time()

    for body_id in range(args.num_bodies):
        output_path = os.path.join(args.output_dir, '%06d' % (body_id))
        if os.path.exists(output_path):
            print('Warnings: The folder %s already exists.' % (output_path))
        else:
            os.makedirs(output_path)

        print('Generating object (%d / %d) at %s...'
              % (body_id, args.num_bodies, output_path))
        body_generator.generate(output_path)

    toc = time.time()
    print('Finished in %.2f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
