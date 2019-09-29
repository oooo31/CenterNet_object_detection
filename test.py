#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
from data.coco import COCO
from progress.bar import IncrementalBar as Bar
import torch
import torch.utils.data as data
from utils.opts import opts
from utils.logger import Logger
from utils.utils import AverageMeter
from models.ctdet import CtdetDetector


class PrefetchDataset(data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        images, meta = {}, {}
        for scale in opt.test_scales:
            images[scale], meta[scale] = self.pre_process_func(image, scale)
        return img_id, {'images': images, 'image': image, 'meta': meta}

    def __len__(self):
        return len(self.images)


def prefetch_test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt = opts().update_dataset_info_and_set_heads(opt, COCO)
    print(opt)
    Logger(opt)

    split = 'val' if not opt.trainval else 'test'
    dataset = COCO(opt, split)
    detector = CtdetDetector(opt)

    data_loader = torch.utils.data.DataLoader(PrefetchDataset(opt, dataset, detector.pre_process),
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)

    results = {}
    num_iters = len(dataset)
    bar = Bar(f'{opt.exp_id}', max=num_iters)
    time_stats = ['tot', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}

    for i, (img_id, pre_processed_images) in enumerate(data_loader):
        ret = detector.run(pre_processed_images)
        results[img_id.numpy().astype(np.int32)[0]] = ret['results']
        bar.suffix = f'{i}/{num_iters}|Elapsed: {bar.elapsed_td} |ETA: {bar.eta_td} '
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            bar.suffix = bar.suffix + '|{} {tm.val:.3f} ({tm.avg:.3f}) '.format(t, tm=avg_time_stats[t])
        bar.next()
    bar.finish()
    dataset.run_eval(results, opt.save_dir)


if __name__ == '__main__':
    opt = opts().parse()
    prefetch_test(opt)
