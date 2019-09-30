import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import json
import numpy as np
import cv2
import os
from utils.image import color_aug, get_border, coco2x1y1x2y2
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian
import math
import cfg
import pdb
import torch.utils.data as data

class COCO(data.Dataset):
    def __init__(self, opt, split):
        super().__init__()
        self.data_dir = os.path.join(opt.data_dir, 'coco')
        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))

        if split == 'test':
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'image_info_test-dev2017.json').format(split)
        else:
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_{}2017.json').format(split)

        self.num_classes = 80
        self.class_name = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                           'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                           'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                           'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                           'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
                           'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                           'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                           'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                           'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self._valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                           24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                           37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                           48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                           58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                           72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                           82, 84, 85, 86, 87, 88, 89, 90]

        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352],
                                  [-0.5832747, 0.00994535, -0.81221408],
                                  [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)

        self.split = split
        self.opt = opt

        print(f'.........Loading coco {split}2017 data.')
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

    @staticmethod
    def _to_float(x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {"image_id": int(image_id),
                                 "category_id": int(category_id),
                                 "bbox": bbox_out,
                                 "score": float("{:.2f}".format(score))}

                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def run_eval(self, results, save_dir):
        json.dump(self.convert_eval_format(results), open('{}/results.json'.format(save_dir), 'w'))
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), cfg.max_objs)

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

        if self.split == 'train':
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w
        else:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)

        flipped = False
        if self.split == 'train':
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = get_border(128, img.shape[1])
            h_border = get_border(128, img.shape[0])

            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_matrix = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_matrix, (input_w, input_h), flags=cv2.INTER_LINEAR)
        inp = inp.astype(np.float32) / 255.

        # TODO:inp appears numbers below 0 after color_aug
        if self.split == 'train':
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        inp = (inp - cfg.mean) / cfg.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // cfg.down_ratio
        output_w = input_w // cfg.down_ratio
        trans_matrix = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((cfg.max_objs, 2), dtype=np.float32)
        reg = np.zeros((cfg.max_objs, 2), dtype=np.float32)
        ind = np.zeros(cfg.max_objs, dtype=np.int64)
        reg_mask = np.zeros(cfg.max_objs, dtype=np.uint8)

        gt_det = []
        for i in range(num_objs):
            ann = anns[i]
            bbox = coco2x1y1x2y2(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_matrix)
            bbox[2:] = affine_transform(bbox[2:], trans_matrix)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                # get an object size-adapative radius
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                draw_umich_gaussian(hm[cls_id], ct_int, radius)
                # 没有返回值，heatmap也没出现在等号左边，hm[cls_id]如何被改变？
                wh[i] = 1. * w, 1. * h
                ind[i] = ct_int[1] * output_w + ct_int[0]
                reg[i] = ct - ct_int
                reg_mask[i] = 1

                gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret = {'input': inp, 'hm': hm, 'reg': reg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}

        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret
