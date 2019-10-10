import os
import time
import sys
import cfg
import torch
import tensorboardX


class Logger:
    def __init__(self):
        """Create a summary writer logging to log_dir."""
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        log_path = cfg.log_dir + f'/{time_str}'
        os.makedirs(log_path)
        file_name = os.path.join(log_path, 'log.txt')

        with open(file_name, 'wt') as log_file:
            log_file.write('torch version: {}\n'.format(torch.__version__))
            log_file.write('cudnn version: {}\n'.format(torch.backends.cudnn.version()))
            log_file.write(f'Cmd:{str(sys.argv)}\n')

            # log_file.write('\n==> Opt:\n')
            # for k, v in sorted(args.items()):
            #     log_file.write('  %s: %s\n' % (str(k), str(v)))

        self.writer = tensorboardX.SummaryWriter(log_dir=log_path)
        self.log = open(log_path + '/log.txt', 'w')

        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write(f'{time_str}: {txt}')
        else:
            self.log.write(txt)

        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()

    def close(self):
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)
