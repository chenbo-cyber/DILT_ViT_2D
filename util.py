import os
import torch
import errno
import train
# import wandb
import numpy as np
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

def up_img(epoch=None):

    val_input = np.load("./dataset/val_input.npy")
    val_label = np.load("./dataset/val_label.npy")

    with torch.no_grad():
        checkpoint = torch.load('./checkpoint/experiment_name/drsn/epoch_{num}.pth'.format(num=epoch), map_location=torch.device('cuda') )
        args = checkpoint['args']
        module = train.set_fr_module(args)
        module.load_state_dict(checkpoint['model'])
        module.cpu()
        module.eval()

    sample = 479
    test_input = np.reshape(val_input[sample], (1, 32))
    test_input = torch.tensor(test_input).view(1, -1, 32)
    test_input = test_input.to(torch.float32)

    test_out = module(test_input)
    test_out = test_out.cpu().detach().numpy()

    data = [[x, y] for (x, y) in zip(np.arange(len(val_label[sample])), val_label[sample])]
    table = wandb.Table(data=data, columns=["train_label_D", "train_label_amp"])
    wandb.log({"train_label_1" : wandb.plot.line(table, "train_label_D", "train_label_amp", title="Train_label")})

    data = [[x, y] for (x, y) in zip(np.arange(len(val_label[sample])), test_out[0, :])]
    table = wandb.Table(data=data, columns=["test_out_D", "test_out_amp"])
    wandb.log({'epoch_{num}'.format(num=epoch) : wandb.plot.line(table, "test_out_D", "test_out_amp", 
                                                                    title='epoch_{num}'.format(num=epoch))})

def model_parameters(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    return num_params


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(model, optimizer, scheduler, args, epoch, experiments_root):
    checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args': args,
    }
    checkpoints_root = os.path.join(experiments_root, 'checkpoints')
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if not os.path.exists(checkpoints_root):
        os.makedirs(checkpoints_root)
    cp = os.path.join(checkpoints_root, 'last.pth')
    fn = os.path.join(checkpoints_root, 'epoch_'+str(epoch)+'.pth')
    torch.save(checkpoint, fn)
    symlink_force(fn, cp)


def load(fn, module_type, device = torch.device('cuda')):
    checkpoint = torch.load(fn, map_location=device)
    args = checkpoint['args']
    if device == torch.device('cpu'):
        args.use_cuda = False
    if module_type == 'fr':
        model = modules.set_fr_module(args)
    elif module_type == 'fc':
        model = modules.set_fc_module(args)
    else:
        raise NotImplementedError('Module type not recognized')
    model.load_state_dict(checkpoint['model'])
    optimizer, scheduler = set_optim(args, model, module_type)
    if checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, scheduler, args, checkpoint['epoch']


def set_optim(args, module, module_type):
    if module_type == 'fr':
        optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_fr)
    elif module_type == 'skip':
        optimizer = torch.optim.RMSprop(module.parameters(), lr=args.lr_fr, alpha=0.9)
    elif module_type == 'fc':
        optimizer = torch.optim.Adam(module.parameters(), lr=args.lr_fc)
    else:
        raise(ValueError('Expected module_type to be fr or fc but got {}'.format(module_type)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, verbose=True, T_mult=2)
    return optimizer, scheduler


def print_args(logger, args, experiments_root):
    message = ''
    for k, v in sorted(vars(args).items()):
        message += '\n{:>30}: {:<30}'.format(str(k), str(v))
    logger.info(message)

    args_path = os.path.join(experiments_root, 'run.args')
    with open(args_path, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')
