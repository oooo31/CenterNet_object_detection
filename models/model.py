import torch
import torch.nn as nn

from models.dla_dcn import get_net as dla_dcn
from models.resnet_dcn import get_net as resnet_dcn
from models.large_hourglass import get_net as hourglass

_model_factory = {'dla': dla_dcn,
                  'resdcn': resnet_dcn,
                  'hourglass': hourglass}


def create_model(backbone, heads, head_conv):
    print('Creating model...')
    num_layers = int(backbone.split('_')[-1]) if '_' in backbone else 0
    name = backbone.split('_')[0] if '_' in backbone else backbone
    model = _model_factory[name]
    model = model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model


def load_model(model, model_path, optimizer=None, lr=None, lr_step=None):
    if model_path == 'last':
        model_path = 'checkpoints/model_last.pth'

    # load the gpu trained model to cpu
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    epoch = checkpoint['epoch']
    print(f'loaded {model_path}, epoch {epoch}')
    state_dict_ = checkpoint['state_dict']
    pth_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            pth_dict[k[7:]] = state_dict_[k]
        else:
            pth_dict[k] = state_dict_[k]

    # check loaded parameters and created model parameters
    model_dict = model.state_dict()
    for k in pth_dict:
        if k in model_dict:
            if pth_dict[k].shape != model_dict[k].shape:
                print(f'Skip parameter {k}, required shape {model_dict[k].shape}, get shape {pth_dict[k].shape}.')
                pth_dict[k] = model_dict[k]
        else:
            print(f'Drop parameter {k}.')

    for k in model_dict:
        if not (k in pth_dict):
            print(f'No param {k}.')
            pth_dict[k] = model_dict[k]

    model.load_state_dict(pth_dict, strict=False)

    # resume optimizer parameters
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        start_lr = lr
        for step in lr_step:
            if start_epoch >= step:
                start_lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = start_lr
        print(f'Resumed optimizer with lr: {start_lr}.')
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}
    if optimizer:
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
