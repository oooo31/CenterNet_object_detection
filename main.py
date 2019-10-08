import os
import torch
import torch.utils.data as data
from utils.opts import opts
from models.model import create_model, load_model, save_model
from utils.logger import Logger
from data.coco import COCO
from trainer import CtdetTrainer
import pdb

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark
    opt = opts.update_dataset_info_and_set_heads(opt, COCO)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpus)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0

    if opt.resume:
        model, optimizer, start_epoch = load_model(model, opt.resume, optimizer, opt.lr, opt.lr_step)

    trainer = CtdetTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

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


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
