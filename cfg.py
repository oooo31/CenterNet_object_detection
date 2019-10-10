#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import numpy as np

init_lr = 1.25e-4
lr_step = [90, 120]
down_ratio = 4
train_resolution = [512, 512]
num_classes = 80
heads = {'hm': num_classes, 'wh': 2, 'reg': 2}
mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)
max_objs = 100
hm_weight = 1
wh_weight = 0.1
off_weight = 1

root_dir = '/home/feiyu/center_net/'
data_dir = os.path.join(root_dir, 'data')
save_dir = os.path.join(root_dir, 'checkpoints')
log_dir = os.path.join(root_dir, 'logs')
debug_dir = os.path.join(root_dir, 'debug')
dirs = [data_dir, save_dir, log_dir, debug_dir]
for path in dirs:
    if not os.path.exists(path):
        os.makedirs(path)