#!/usr/bin/env python

"""Generate bodies.
"""

import argparse
import os
import json
import bodies


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--body',
                        dest='body_type',
                        help='The body type to genearte.',
                        type=str,
                        required=True)

    parser.add_argument('--config',
                        dest='config_json',
                        help='The json config file.',
                        type=str,
                        required=True)

    parser.add_argument('--output',
                        dest='output_dir',
                        help='The output directory.',
                        type=str,
                        required=True)

    parser.add_argument('--color',
                        dest='color',
                        help='Choose wood or metal',
                        type=str,
                        required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.body_type == 'hammer':
        body_class = bodies.SimpleHammer
    elif args.body_type == 'part':
        body_class = bodies.SimplePart
    else:
        raise ValueError

    config = json.load(open(args.config_json, 'r'))

    body_generator = body_class(name='body',
                                config=config,
                                color=args.color)

    output_path = args.output_dir
    body_generator.generate(output_path)


if __name__ == '__main__':
    main()
