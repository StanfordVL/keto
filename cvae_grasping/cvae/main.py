import os
import argparse

from cvae.build import train_vae, train_gcnn
from cvae.build import inference_vae_gcnn

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

if args.train == 'vae':
    train_vae(data_path=args.data_path,
              model_path=args.model_path)
elif args.train == 'gcnn':
    train_gcnn(data_path=args.data_path,
               model_path=args.model_path)
elif args.train == 'test':
    inference_vae_gcnn(
               data_path=args.data_path,
               model_path=args.model_path)
else:
    raise NotImplementedError
