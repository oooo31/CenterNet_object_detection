import argparse
import os
import cfg


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--debug', type=int, default=0,
                                 help='level of visualization.'
                                      '1: only show the final detection results'
                                      '2: show the network output features'
                                      '3: use matplot to display'  # useful when lunching training with ipython notebook
                                      '4: save all visualizations to disk')
        self.parser.add_argument('--demo', default='',
                                 help='path to image/ image folders/ video. or "webcam"')


        # system
        self.parser.add_argument('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        # log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                                 help='visualization threshold.')
        self.parser.add_argument('--debugger_theme', default='white',
                                 choices=['white', 'black'])

        # model
        self.parser.add_argument('--arch', default='dla_34',
                                 help='model architecture. Currently tested'
                                      'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                      'dlav0_34 | dla_34 | hourglass')
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '64 for resnets and 256 for dla.')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')

        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4, help='initial learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='90,120', help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=140, help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1, help='batch size on the master gpu.')
        self.parser.add_argument('--val_intervals', type=int, default=2, help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                 help='include validation in training and test on test set')
        # ground truth validation
        self.parser.add_argument('--eval_oracle_hm', action='store_true',
                                 help='use ground center heatmap.')
        self.parser.add_argument('--eval_oracle_wh', action='store_true',
                                 help='use ground truth bounding box size.')
        self.parser.add_argument('--eval_oracle_offset', action='store_true',
                                 help='use ground truth local heatmap offset.')

        # test
        self.parser.add_argument('--flip_test', action='store_true', help='flip data augmentation.')
        self.parser.add_argument('--test_scales', type=str, default='1', help='multi scale test augmentation.')
        self.parser.add_argument('--K', type=int, default=100, help='max number of output objects.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')

        self.parser.add_argument('--flip', type=float, default=0.5,
                                 help='probability of applying flip augmentation.')

        self.parser.add_argument('--resume', default=None, type=str, help='The path of checkpoint file to resume'
                                                                          'training from, \'last\' for model_last.pth.')

    def parse(self):
        opt = self.parser.parse_args()
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 64
        opt.pad = 127 if 'hourglass' in opt.arch else 31
        opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.debug > 0:
            opt.num_workers = 0
            opt.batch_size = 1
            opt.gpus = [opt.gpus[0]]
            opt.master_batch_size = -1

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]

        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = '/home/feiyu/center_net/'
        opt.data_dir = os.path.join(opt.root_dir, 'data')
        opt.exp_dir = os.path.join(opt.root_dir, 'logs')
        opt.save_dir = os.path.join(opt.root_dir, 'checkpoints')
        opt.debug_dir = os.path.join(opt.root_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        return opt

    @staticmethod
    def update_dataset_info_and_set_heads(opt, dataset):
        opt.num_classes = cfg.num_classes

        # assert opt.dataset in ['pascal', 'coco']
        opt.heads = {'hm': opt.num_classes, 'wh': 2, 'reg': 2}

        return opt

    def init(self, args=''):
        default_dataset_info = {'ctdet': {'default_resolution': [512, 512], 'num_classes': 80,
                                          'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                                          'dataset': 'coco'}}

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
