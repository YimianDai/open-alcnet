import os
# os.system("taskset -p -c 40-47 %d" % os.getpid())

import platform
import sys
import socket
import argparse
import numpy as np
from tqdm import tqdm

from datetime import datetime

import mxnet as mx
from mxnet import gluon, autograd, init
from mxnet.gluon.data.vision import transforms

from gluoncv.utils import LRScheduler

from data import IceContrast
from model import MPCMResNetFPN
from metric import SigmoidMetric, SamplewiseSigmoidMetric
from loss import SoftIoULoss, SamplewiseSoftIoULoss

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model
    parser.add_argument('--net-choice', type=str, default='PCMNet',
                        help='model name PCMNet, PlainNet')
    parser.add_argument('--pyramid-mode', type=str, default='Dec',
                        help='Inc, Dec')
    parser.add_argument('--r', type=int, default=2, help='1, 2, 4')
    parser.add_argument('--summary', action='store_true',
                        help='print parameters')
    parser.add_argument('--scale-mode', type=str, default='xxx',
                        help='Single, Multiple, Selective')
    parser.add_argument('--pyramid-fuse', type=str, default='sk',
                        help='add, max, sk')
    parser.add_argument('--cue', type=str, default='lcm', help='lcm or orig')
    # dataset
    parser.add_argument('--dataset', type=str, default='DENTIST',
                        help='dataset name (default: DENTIST, Iceberg)')
    parser.add_argument('--workers', type=int, default=48,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--blocks', type=int, default=4,
                        help='[1] * blocks')
    parser.add_argument('--channels', type=int, default=16,
                        help='channels')
    parser.add_argument('--shift', type=int, default=13,
                        help='shift')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='iou-thresh')
    parser.add_argument('--crop-size', type=int, default=480,
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
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=32,
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
    # parser.add_argument('--no-cuda', action='store_true', default=
    #                     True, help='disables CUDA training')
    # parser.add_argument('--ngpus', type=int,
    #                     default=len(mx.test_utils.list_gpus()),
    #                     help='number of GPUs (default: 4)')
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
        # print('Number of GPUs:', args.ngpus)
        # args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
        args.ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        print('Number of GPUs:', len(args.ctx))

    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    # args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    args.norm_kwargs = {'num_devices': len(args.ctx)} if args.syncbn else {}
    print(args)
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]), # Default mean and std
            # transforms.Normalize([.418, .447, .571], [.091, .078, .076]),   # Iceberg mean and std
        ])
        ################################# dataset and dataloader #################################
        if platform.system() == "Darwin":
            data_root = os.path.join('~', 'Nutstore Files', 'Dataset')
        elif platform.system() == "Linux":
            data_root = os.path.join('~', 'datasets')
            if args.colab:
                # data_root = '/content/gdrive/My Drive/Colab Notebooks/datasets'
                data_root = '/content/datasets'
        else:
            raise ValueError('Notice Dataset Path')

        data_kwargs = {'base_size': args.base_size, 'transform': input_transform,
                       'crop_size': args.crop_size, 'root': data_root,
                       'base_dir' : args.dataset}
        trainset = IceContrast(split=args.train_split, mode='train',   **data_kwargs)
        valset = IceContrast(split=args.val_split, mode='testval', **data_kwargs)

        self.train_data = gluon.data.DataLoader(trainset, args.batch_size, shuffle=True,
            last_batch='rollover', num_workers=args.workers)
        self.eval_data = gluon.data.DataLoader(valset, args.test_batch_size,
            last_batch='rollover', num_workers=args.workers)

        model = '' #Fix scope issue

        # net_choice = 'PCMNet'  # ResNetFPN, PCMNet, MPCMNet, LayerwiseMPCMNet
        net_choice = self.args.net_choice
        print("net_choice: ", net_choice)

        if net_choice == 'MPCMResNetFPN':
            r = self.args.r
            layers = [self.args.blocks] * 3
            channels = [8, 16, 32, 64]
            shift = self.args.shift
            pyramid_mode = self.args.pyramid_mode
            scale_mode = self.args.scale_mode
            pyramid_fuse = self.args.pyramid_fuse

            model = MPCMResNetFPN(layers=layers, channels=channels, shift=shift,
                                  pyramid_mode=pyramid_mode, scale_mode=scale_mode,
                                  pyramid_fuse=pyramid_fuse, r=r, classes=trainset.NUM_CLASS)
            print("net_choice: ", net_choice)
            print("scale_mode: ", scale_mode)
            print("pyramid_fuse: ", pyramid_fuse)
            print("r: ", r)
            print("layers: ", layers)
            print("channels: ", channels)
            print("shift: ", shift)


        self.host_name = socket.gethostname()
        self.save_prefix = self.host_name + '_' + net_choice + '_scale-mode_' + args.scale_mode + \
                           '_pyramid-fuse_' + args.pyramid_fuse + '_b_' + str(args.blocks)
        if args.net_choice == 'ResNetFCN':
            self.save_prefix = self.host_name + '_' + net_choice + '_b_' + str(args.blocks)

        # resume checkpoint if needed
        if args.resume is not None:
            if os.path.isfile(args.resume):
                model.load_parameters(args.resume, ctx=args.ctx)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        else:
            # model.initialize(init=init.Xavier(), ctx=args.ctx, force_reinit=True)
            model.initialize(init=init.MSRAPrelu(), ctx=args.ctx, force_reinit=True)
            print("Model Initializing")
            print("args.ctx: ", args.ctx)


        self.net = model
        # self.net.summary(mx.nd.zeros((1, 3, 480, 480)))

        if args.summary:
            self.net.summary(mx.nd.zeros((1, 3, 480, 480), self.args.ctx[0]))
            sys.exit()

        # create criterion
        self.criterion = SoftIoULoss()

        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr,
                                        nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_data),
                                        power=0.9)
        kv = mx.kv.create(args.kvstore)

        # For SGD
        # optimizer_params = {'lr_scheduler': self.lr_scheduler,
        #                     'wd': args.weight_decay,
        #                     'momentum': args.momentum,
        #                     'learning_rate': args.lr
        #                    }
        optimizer_params = {
            # 'lr_scheduler': self.lr_scheduler,
            'wd': args.weight_decay,
            'learning_rate': args.lr
        }
        # For Adam

        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        if args.no_wd:
            for k, v in self.net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        # self.optimizer = gluon.Trainer(self.net.collect_params(), 'sgd',
        #                                optimizer_params, kvstore = kv)
        # self.optimizer = gluon.Trainer(self.net.collect_params(), 'adam',
        #                                optimizer_params, kvstore = kv)
        self.optimizer = gluon.Trainer(self.net.collect_params(), 'adagrad',
                                       optimizer_params, kvstore=kv)
        # self.optimizer = gluon.Trainer(self.net.collect_params(), 'nag',
        #                                optimizer_params, kvstore=kv)

        ################################# evaluation metrics #################################

        self.iou_metric = SigmoidMetric(1)
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=self.args.score_thresh)
        # self.metric = Seg2DetVOC07MApMetric(iou_thresh=self.args.iou_thresh,
        #                                     sparsity=self.args.sparsity,
        #                                     score_thresh=self.args.score_thresh)
        self.best_metric = 0
        self.best_iou = 0
        self.best_nIoU = 0
        self.is_best = False

    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.0
        for i, batch in enumerate(tbar):
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.args.ctx, batch_axis=0)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=self.args.ctx, batch_axis=0)
            losses = []
            with autograd.record(True):
                for x, y in zip(data, labels):
                    pred = self.net(x)
                    loss = self.criterion(pred, y)
                    losses.append(loss)
                mx.nd.waitall()
                autograd.backward(losses)
            self.optimizer.step(self.args.batch_size)
            for loss in losses:
                train_loss += np.mean(loss.asnumpy()) / len(losses)
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, train_loss/(i+1)))

    def validation(self, epoch):
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        # self.metric.reset()
        tbar = tqdm(self.eval_data)
        for i, batch in enumerate(tbar):
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.args.ctx, batch_axis=0)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=self.args.ctx, batch_axis=0)
            preds = []
            for x, y in zip(data, labels):
                pred = self.net(x)
                preds.append(pred)
            # self.metric.update(preds, labels)
            self.iou_metric.update(preds, labels)
            self.nIoU_metric.update(preds, labels)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            tbar.set_description('Epoch %d, IoU: %.4f, nIoU: %.4f' % (epoch, IoU, nIoU))

        if IoU > self.best_iou:
            self.best_iou = IoU
            self.net.save_parameters('tmp_{:s}_best_{:s}.params'.format(
                self.save_prefix, 'IoU'))
            with open(self.save_prefix + '_GPU_' + self.args.gpus +
                      '_best_IoU.log', 'a') as f:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epoch, IoU))

        if nIoU > self.best_nIoU:
            self.best_nIoU = nIoU
            self.net.save_parameters('tmp_{:s}_best_{:s}.params'.format(
                self.save_prefix, 'nIoU'))
            with open(self.save_prefix + '_GPU_' + self.args.gpus +
                      '_best_nIoU.log', 'a') as f:
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epoch, nIoU))

        if epoch >= args.epochs - 1:
            print("best_iou: ", self.best_iou)
            print("best_nIoU: ", self.best_nIoU)

        if epoch >= args.epochs - 1:
            with open(self.save_prefix + '_' + '_GPU_' + self.args.gpus +
                      '_best_IoU.log', 'a') as f:
                f.write('Finished\n')
            self.net.save_parameters('tmp_{:s}_best_{:s}_{:s}.params'.format(
                self.save_prefix, 'IoU', str(self.best_iou)))
            self.net.save_parameters('tmp_{:s}_best_{:s}_{:s}.params'.format(
                self.save_prefix, 'nIoU', str(self.best_nIoU)))


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epochs:', args.epochs)
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                trainer.validation(epoch)
