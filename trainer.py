import torch
import numpy as np
import time
import pdb
from progress.bar import Bar
import torch.nn as nn

import cfg
from models.losses import FocalLoss
from models.losses import RegL1Loss
from models.decode import ctdet_decode
from utils.utils import _sigmoid
from utils.debugger import Debugger
from utils.oracle_utils import gen_oracle_map
from models.data_parallel import DataParallel
from utils.utils import AverageMeter


class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class CtdetLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0

        for s in range(opt.num_stacks):
            output = outputs[s]

            output['hm'] = _sigmoid(output['hm'])

            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(
                    gen_oracle_map(batch['wh'].detach().cpu().numpy(),
                                   batch['ind'].detach().cpu().numpy(),
                                   output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)

            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(
                    gen_oracle_map(batch['reg'].detach().cpu().numpy(),
                                   batch['ind'].detach().cpu().numpy(),
                                   output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += FocalLoss(output['hm'], batch['hm']) / opt.num_stacks
            wh_loss += RegL1Loss(output['wh'], batch['reg_mask'], batch['ind'], batch['wh']) / opt.num_stacks
            off_loss += RegL1Loss(output['reg'], batch['reg_mask'], batch['ind'], batch['reg']) / opt.num_stacks

        loss = cfg.hm_weight * hm_loss + cfg.wh_weight * wh_loss + cfg.off_weight * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


class CtdetTrainer:
    def __init__(self, opt, model, optimizer):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        self.loss = CtdetLoss(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(output['hm'], output['wh'], reg=reg, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio

        for i in range(1):
            debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')

            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1], dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1], dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(self.model_with_loss, device_ids=gpus, chunk_sizes=chunk_sizes).to(
                device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_states}
        num_iters = len(data_loader)
        bar = Bar(f'{opt.exp_id}', max=num_iters)
        end = time.time()

        for i, batch in enumerate(data_loader):
            if i >= num_iters:
                break

            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = f'{phase}: [{epoch}][{i}/{num_iters}]|Tot: {bar.elapsed_td:} |ETA: {bar.eta_td:} '

            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
                bar.suffix = bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            bar.suffix = bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                      '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

            bar.next()

            if opt.debug > 0:
                self.debug(batch, output, i)

            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.

        return ret, results
