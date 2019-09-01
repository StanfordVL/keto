import os
import argparse

from cvae.build import train_vae_grasp, train_gcnn_grasp
from cvae.build import train_vae_keypoint, train_discr_keypoint
from cvae.build import train_vae_action, train_discr_action
from cvae.build import inference_grasp, inference_keypoint

parser = argparse.ArgumentParser()

parser.add_argument(
    '--train',
    type=str,
    default=None,
    help='vae or gcnn or test')

parser.add_argument(
    '--model_path',
    type=str,
    default=None,
    help='pretrained model')

parser.add_argument(
    '--task_name',
    type=str,
    default='task')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='gpu to use')

parser.add_argument(
    '--data_path',
    type=str,
    default='./data/data.hdf5',
    help='path to data in hdf5 format')


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.train == 'vae_grasp':
    train_vae_grasp(data_path=args.data_path,
                    model_path=args.model_path)
elif args.train == 'gcnn_grasp':
    train_gcnn_grasp(data_path=args.data_path,
                     model_path=args.model_path)
elif args.train == 'inference_grasp':
    inference_grasp(
        data_path=args.data_path,
        model_path=args.model_path)
elif args.train == 'vae_keypoint':
    train_vae_keypoint(data_path=args.data_path,
                       model_path=args.model_path,
                       task_name=args.task_name)
elif args.train == 'discr_keypoint':
    train_discr_keypoint(data_path=args.data_path,
                         model_path=args.model_path,
                         task_name=args.task_name)
elif args.train == 'vae_action':
    train_vae_action(data_path=args.data_path, 
                     model_path=args.model_path,
                     task_name=args.task_name)
elif args.train == 'discr_action':
    train_discr_action(data_path=args.data_path, 
                       model_path=args.model_path,
                       task_name=args.task_name)
elif args.train == 'inference_keypoint':
    inference_keypoint(
        data_path=args.data_path,
        model_path=args.model_path)

else:
    raise NotImplementedError
