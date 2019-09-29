#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import cv2

aa = cv2.imread('images/3a6e79a177a69fbebec1599a3b8fcd2e.jpg')
# aa = aa[:, :, ::-1]
cv2.imshow('aa', aa)
cv2.waitKey()