import os
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('--script',
                    type=str,
                    required=True)
parser.add_argument('--num_copies',
                    type=int,
                    required=True)

args = parser.parse_args()

command = "sh {}".format(args.script)

for iprocess in range(args.num_copies):
    print('Calling process {}'.format(iprocess))
    subprocess.Popen(command, shell=True)

while True:
    pass
