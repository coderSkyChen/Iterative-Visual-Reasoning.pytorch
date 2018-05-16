# @Time    : 2018/3/27 8:37
# @File    : opts.py.py
# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn

import argparse
import numpy as np


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--DATA_DIR', help='root directory for dataset',
                        default='/home/mcg/cxk/iter-reason/data')
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--caffe', help='use caffe pretrained model, path of model',
                        type=str, default=None)
    # parser.add_argument('--with_global', help='whether use the global module',
    #                     type=bool, default=False)
    parser.add_argument('--MEM_ITER', help='num of iter', type=int, default=2)
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='ade', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20000, type=int)

    parser.add_argument('--iters', dest='max_iters',
                        help='number of iters to train',
                        default=320000, type=int)

    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=5, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="./models",
                        nargs=argparse.REMAINDER)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=6, type=int)
    parser.add_argument('--cuda', type=bool, default=False)

    parser.add_argument('--mGPUs', default=False, type=bool)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size should always be 1',
                        default=1, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is iters',
                        default=280000, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--root_log', type=str, default='log')
    parser.add_argument('--root_model', type=str, default='model')
    parser.add_argument('--root_output', type=str, default='output')
    # set training session
    parser.add_argument('--train_id',
                        help='train_id is used to identify this training phrase', type=str)

    # resume trained model
    parser.add_argument('--resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--model_name', type=str, help='modelfile name,relative path for the model file')

    parser.add_argument('--vis', default=False, type=bool)

    args = parser.parse_args()

    args.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    args.EPS = 1e-14
    args.MOMENTUM = 0.9
    args.BOTTLE_SCALE = 16.
    args.TRUNCATED = False
    args.DOUBLE_BIAS = True
    args.BIAS_DECAY = False
    args.WEIGHT_DECAY = 0.0001
    args.RNG_SEED = 3
    args.POOLING_SIZE = 7
    args.FEAT_STRIDE = [16, ]

    args.MEM_C = 512
    # args.MEM_ITER = 2
    args.MEM_FC_L = 2
    args.MEM_IN_CONV = 3
    args.MEM_BETA = 0.5
    return args
