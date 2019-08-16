import os
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--command',
                    type=str,
                    required=True)
parser.add_argument('--process',
                    type=int,
                    required=True)

args = parser.parse_args()

for iprocess in range(args.process):
    print('Calling process {}'.format(iprocess))
    subprocess.Popen(args.command, shell=True)
