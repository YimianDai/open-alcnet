import os
# os.system("taskset -p -c 1-96 %d" % os.getpid())
import scipy.misc
import platform
import timeit
import sys
import socket
import argparse
import numpy as np
from utils import summary
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

from data import IceContrast
from model import MPCMResNetFPN
from loss import SoftIoULoss

import matplotlib.pyplot as plt

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model
    parser.add_argument('--net-choice', type=str, default='MPCMResNetFPN',
                        help='model name PCMNet, PlainNet')
    parser.add_argument('--pyramid-mode', type=str, default='Dec',
                        help='Inc, Dec')
    parser.add_argument('--scale-mode', type=str, default='Multiple',
                        help='Single, Multiple, Selective')
    parser.add_argument('--pyramid-fuse', type=str, default='bottomuplocal',
                        help='add, max, sk')
    parser.add_argument('--cue', type=str, default='lcm', help='lcm or orig')
    # dataset
    parser.add_argument('--dataset', type=str, default='DENTIST',
                        help='dataset name (default: DENTIST, Iceberg)')
    parser.add_argument('--workers', type=int, default=48,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--blocks', type=int, default=4,
                        help='[1] * blocks')
    parser.add_argument('--channels', type=int, default=16,
                        help='channels')
    parser.add_argument('--shift', type=int, default=13,
                        help='shift')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='iou-thresh')
    parser.add_argument('--crop-size', type=int, default=240,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='trainval',
                        help='dataset train split (default: train)')
    parser.add_argument('--val-split', type=str, default='test',
                        help='dataset val split (default: val)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='100,200',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--gamma', type=int, default=2,
                        help='gamma for Focal Soft IoU Loss')
    parser.add_argument('--lambd', type=int, default=1,
                        help='lambd for TV Soft IoU Loss')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    parser.add_argument('--sparsity', action='store_true', default=
                        False, help='')
    parser.add_argument('--score-thresh', type=float, default=0.5,
                        help='score-thresh')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--colab', action='store_true', default=
                        False, help='whether using colab')

    # evaluation only
    parser.add_argument('--eval', action='store_true', default= False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
    parser.add_argument('--metric', type=str, default='mAP',
                        help='F1, IoU, mAP')

    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')

    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.no_cuda or (len(mx.test_utils.list_gpus()) == 0):
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        args.ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        print('Number of GPUs:', len(args.ctx))

    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': len(args.ctx)} if args.syncbn else {}
    print(args)
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
        ])
        ################################# dataset and dataloader #################################
        if platform.system() == "Darwin":
            data_root = os.path.join('~', 'Nutstore Files', 'Dataset')
        elif platform.system() == "Linux":
            data_root = os.path.join('~', 'datasets')
            if args.colab:
                data_root = '/content/datasets'
        else:
            raise ValueError('Notice Dataset Path')

        data_kwargs = {'base_size': args.base_size, 'transform': input_transform,
                       'crop_size': args.crop_size, 'root': data_root,
                       'base_dir' : args.dataset}
        # data_kwargs = {'base_size': args.base_size,
        #                'crop_size': args.crop_size, 'root': data_root,
        #                'base_dir' : args.dataset}
        valset = IceContrast(split=args.val_split, mode='testval', include_name=True,
                             **data_kwargs)
        self.valset = valset

        net_choice = args.net_choice
        print("net_choice: ", net_choice)

        if net_choice == 'MPCMResNetFPN':
            layers = [self.args.blocks] * 3
            channels = [8, 16, 32, 64]
            shift = self.args.shift
            pyramid_mode = self.args.pyramid_mode
            scale_mode = self.args.scale_mode
            pyramid_fuse = self.args.pyramid_fuse

            model = MPCMResNetFPN(layers=layers, channels=channels, shift=shift,
                                  pyramid_mode=pyramid_mode, scale_mode=scale_mode,
                                  pyramid_fuse=pyramid_fuse, classes=valset.NUM_CLASS)
            print("net_choice: ", net_choice)
            print("scale_mode: ", scale_mode)
            print("pyramid_fuse: ", pyramid_fuse)
            print("layers: ", layers)
            print("channels: ", channels)
            print("shift: ", shift)
        else:
            raise ValueError('Unknow net_choice')

        self.host_name = socket.gethostname()
        self.save_prefix = 'MLCPFN' + '_' + args.scale_mode + '_' + args.pyramid_fuse + '_'

        params_path = './params/BottomUpLocal_r_1_b_4_0.7614.params'
        model.load_parameters(params_path, ctx=args.ctx)
        self.net = model

        # create criterion
        kv = mx.kv.create(args.kvstore)

        optimizer_params = {
            'wd': args.weight_decay,
            'learning_rate': args.lr
        }

        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        if args.no_wd:
            for k, v in self.net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

    ################################# evaluation metrics #################################

    def validation(self, epoch):
        save_path = os.path.expanduser('/Users/grok/Downloads/')

        # summary(self.net, mx.nd.zeros((1, 3, args.crop_size, args.crop_size), ctx=args.ctx[0]))
        # sys.exit()

        i = 0
        mx.nd.waitall()
        start = timeit.default_timer()
        for img, mask, img_id in self.valset:
            exp_img = img.expand_dims(axis=0)
            if not (len(mx.test_utils.list_gpus()) == 0):
                exp_img = exp_img.copyto(mx.gpu())
            # pred = self.net(exp_img).squeeze().asnumpy() > 0
            pred = self.net(exp_img)
            plt.imsave(save_path + img_id + '.png', pred)
            # print(pred.shape)

        # save_path = os.path.expanduser('/Users/grok/Downloads/img')
        # for img, mask, img_id in self.valset:
            # exp_img = img.expand_dims(axis=0)
            # img = mx.nd.transpose(img, (1, 2, 0))
            # print(img.shape)
            # img = img.squeeze().asnumpy() / 255
            # plt.imsave(save_path + img_id + '.png', img)


            # break




if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epochs:', args.epochs)
        trainer.validation(0)
