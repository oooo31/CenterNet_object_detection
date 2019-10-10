#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import cv2
import cfg
import numpy as np
from data.coco import COCO
from progress.bar import IncrementalBar as Bar
import torch
import torch.utils.data as data
from utils.image import get_affine_transform
from utils.opts import opts
from utils.logger import Logger
from utils.utils import AverageMeter
from models.ctdet import CtdetDetector


def pre_process(self, image, scale):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height = (new_height | self.opt.pad) + 1  # a different way of padding
    inp_width = (new_width | self.opt.pad) + 1

    c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
    s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_matrix = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(resized_image, trans_matrix, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - cfg.mean) / cfg.std).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

    if self.opt.flip_test:
        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)

    meta = {'c': c, 's': s, 'out_height': inp_height // cfg.down_ratio, 'out_width': inp_width // cfg.down_ratio}

    return images, meta


class PrefetchDataset(data.Dataset):
    def __init__(self, opt, dataset, pre_process):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process = pre_process
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        images, meta = {}, {}

        for scale in opt.test_scales:
            images[scale], meta[scale] = self.pre_process(image, scale)
        return img_id, {'images': images, 'image': image, 'meta': meta}

    def __len__(self):
        return len(self.images)


def prefetch_test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpus)
    opt = opts().update_dataset_info_and_set_heads(opt, COCO)
    print(opt)
    Logger(opt)

    split = 'val' if not opt.trainval else 'test'
    dataset = COCO(opt, split)
    detector = CtdetDetector(opt)

    data_loader = torch.utils.data.DataLoader(PrefetchDataset(opt, dataset, pre_process),
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)

    results = {}
    num_iters = len(dataset)
    bar = Bar(f'{opt.exp_id}', max=num_iters)
    time_stats = ['tot', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}

    for i, (img_id, images) in enumerate(data_loader):
        ret = detector.run(images)
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
