import os
import torch
import argparse
import torch.utils.data as data

import cfg
from models.model import create_model, load_model, save_model
from utils.logger import Logger
from data.coco import COCO
from trainer import CtdetTrainer
import pdb

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Training CenterNet.')
parser.add_argument('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')
parser.add_argument('--backbone', default='dla_34', help='including res_101, resdcn_101, dla_34, hourglass')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--resume', default=None, type=str, help='The path of checkpoint file to resume training from, '
                                                             '\'last\' for model_last.pth.')
# ground truth validation
parser.add_argument('--eval_gt_hm', action='store_true', help='use ground center heatmap.')
parser.add_argument('--eval_gt_wh', action='store_true', help='use ground truth bounding box size.')
parser.add_argument('--eval_gt_offset', action='store_true', help='use ground truth local heatmap offset.')
args = parser.parse_args()

logger = Logger()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
args.device = torch.device('cuda' if '-1' not in args.gpus else 'cpu')


lr = (args.batch_size / 32) * cfg.init_lr

net = create_model(args.backbone, cfg.heads, head_channel)
optimizer = torch.optim.Adam(net.parameters(), lr)

start_epoch = 0

if args.resume:
    model, optimizer, start_epoch = load_model(model, args.resume, optimizer, lr, cfg.lr_step)

trainer = CtdetTrainer(args, model, optimizer)
trainer.set_device(opt.gpus, opt.chunk_sizes, device)

print('Setting up data...')
val_loader = data.DataLoader(COCO(opt, 'val'), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

train_loader = data.DataLoader(COCO(opt, 'train'),
                               batch_size=opt.batch_size,
                               shuffle=False,
                               num_workers=1,
                               pin_memory=True,
                               drop_last=True)

print('Starting training...')
best = 1e10

for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    log_dict_train, _ = trainer.run_epoch('train', epoch, train_loader)
    save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)

    logger.write(f'epoch: {epoch} |')

    for k, v in log_dict_train.items():
        logger.scalar_summary(f'train_{k}', v, epoch)
        logger.write('{} {:8f} | '.format(k, v))

    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:

        with torch.no_grad():
            log_dict_val, preds = trainer.run_epoch('val', epoch, val_loader)
        for k, v in log_dict_val.items():
            logger.scalar_summary(f'val_{k}', v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if log_dict_val[opt.metric] < best:
            best = log_dict_val[opt.metric]
            save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)

    logger.write('\n')

    if epoch in opt.lr_step:
        lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
        print('Drop LR to', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

logger.close()
