#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np

down_ratio = 4
train_resolution = [512, 512]
num_classes = 80
mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)
max_objs = 128