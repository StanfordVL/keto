import os
import shutil
import argparse

DATA_TYPES = ['point_cloud', 'keypoints']

parser = argparse.ArgumentParser()

parser.add_argument('--data_path',
                    type=str)

parser.add_argument('--output_path',
                    type=str,
                    default='./raw')

parser.add_argument('--task_name',
                    type=str)

args = parser.parse_args()

data_path = args.data_path
output_path = args.output_path
task_name = args.task_name

episodes = [e for e in os.listdir(data_path) if task_name in e]
episodes.sort()

cwd = os.getcwd()

count = -1
for episode in episodes:
    eps_path = os.path.join(cwd, data_path, episode)
    eps_files = os.listdir(os.path.join(eps_path, DATA_TYPES[0]))
    for eps_file in eps_files:
        count = count + 1
        for data_type in DATA_TYPES:
            dst_dir = os.path.join(output_path, task_name, data_type)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            src = os.path.join(eps_path, data_type, eps_file)
            dst = os.path.join(output_path, task_name, data_type,
                    str(count).zfill(6) + '.npy')
            os.symlink(src, dst)

