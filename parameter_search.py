from utils import utils
from train_cls import main
import argparse

import numpy as np


parser = argparse.ArgumentParser(description='Setting for Point Cloud Classification')
parser.add_argument('--exp_name', type=str, default='', metavar='N',
                    help='Name of the experiment')

parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'],
                    help='Model to use, [pointnet, dgcnn]. (Default: dgcnn)')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--data_path', type=str, default='', metavar='N',
                    help='Dataset path')
parser.add_argument('--num_classes', type=int, default=4, metavar='N',
                        help='Number of classes')
parser.add_argument('--training_data', type=str, default=["synthetic", "real"], metavar='N',
                    help='Training data type')
parser.add_argument('--training_mode', type=str, default="multi-class", metavar='N',
                    help='Training mode')

parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of batch. (Default: 32)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='Number of episode to train. (Default: 50)')

parser.add_argument('--use_sgd', type=bool, default=True,
                    help='Use SGD')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='Learning rate. (Default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum. (Default: 0.9)')
parser.add_argument('--wd', type=float, default=0.001, metavar='LR',
                    help='Weight Decay. (Default: 0.001)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate. (Default: 0.5)')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings. (Default: 1024)')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use in DGCNN. (Default: 20)')
parser.add_argument('--transform_regularization', type=float, default=0.001, metavar='N',
                    help='Regularization of the fature transform in PointNet. (Default: 0.001)')


args = parser.parse_args()


lr_list = [1, 3]
weight_decay_list = [1, 5]

Datapaths = [("Scenario1", ["synthetic"]), ("Scenario2", ["real"]), ("Scenario3", ["synthetic", "real"])]

args.data_path = "./data"

for datapath in Datapaths:
    exp_name, args.training_data = datapath

    for lr_x in range(lr_list[0], lr_list[-1]+1):
        for wd_x in range(weight_decay_list[0], weight_decay_list[-1]+1):

            # Get the new hyperparameters
            lr = 10**(-lr_x)
            weight_decay = 10**(-wd_x)

            print(lr_x ,wd_x)

            # Set parameters in the arg dir
            args.wd = weight_decay
            args.lr = lr
            args.exp_name = "{}_{}_{:.6f}_{:.6f}".format(exp_name, args.model, args.lr, args.wd)


            main(args)
        
