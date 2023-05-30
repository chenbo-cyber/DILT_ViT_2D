import os
import sys
import time
import argparse
import logging

import torch
import numpy as np
import scipy.io as scio
# import wandb
import util
import os
import matplotlib.pyplot as plt

import dataset_r as dataset
from Module.SimpleViT import *
from Module.ViT import *
from Module.MAE import *
from Module.EANet import *

# DEVICE = torch.device("cuda:0")
logger = logging.getLogger(__name__)

def set_module(args):
    """
    Create module
    """
    DEVICE = torch.device(args.device)
    # DEVICE = torch.device("cuda:0")
    net = None
    b = np.linspace(0.015, 1.5, 100)[:, np.newaxis]
    t = np.linspace(0.015, 1.5, 100)[:, np.newaxis]
    # b = scio.loadmat('Dataset/T2-T2_mt50ms.mat')['t2values']
    # t = scio.loadmat('Dataset/T2-T2_mt50ms.mat')['t2values']
    if args.module_type == 'EANet':
        net = EANet(n_classes=1, n_layers=50, stride=8, b_values=b, t_values=t, DEVICE=DEVICE)
    elif args.module_type == 'SimpleViT':
        net = SimpleViT(
                        image_size = args.image_size,
                        patch_size = args.patch_size,
                        num_classes = 10000,
                        dim = 1024,
                        depth = args.n_layers,
                        heads = args.n_heads,
                        mlp_dim = 2048,
                        b_values=b, 
                        t_values=t, 
                        DEVICE=DEVICE
                    )
    elif args.module_type == 'ViT':
        net = ViT(
                        image_size = 100,
                        patch_size = 10,
                        num_classes = 10000,
                        dim = 1024,
                        depth = 6,
                        heads = 8,
                        mlp_dim = 2048,
                        dropout = 0.1,
                        emb_dropout = 0.1,
                        b_values=b, 
                        t_values=t, 
                        DEVICE=DEVICE
                    )
    elif args.module_type == 'MAE':
        v = SimpleViT(
                    image_size = 100,
                    patch_size = 10,
                    num_classes = 1000,
                    dim = 1024,
                    depth = 6,
                    heads = 8,
                    mlp_dim = 2048,
                    b_values=b, 
                    t_values=t, 
                    DEVICE=DEVICE
                )

        net = MAE(
                    encoder = v,
                    encoder_dim=1024,
                    masking_ratio = 0.5,   # the paper recommended 75% masked patches
                    decoder_dim = 1024,      # paper showed good results with just 512
                    decoder_depth = 6,       # anywhere from 1 to 8
                    num_classes=10000
                )

    else:
        raise NotImplementedError('Module type not implemented')
    if args.use_cuda:
        net.to(DEVICE)
    return net

def train(args, module, optimizer, criterion, scheduler, train_loader, val_loader, test_loader, epoch, experiments_root):
    """
    Train the module for one epoch
    """
    DEVICE = torch.device(args.device)
    # DEVICE = torch.device("cuda:0")
    epoch_start_time = time.time()
    module.train()
    loss_train = 0
    loss_l1 = torch.nn.L1Loss(reduction='sum')
    for batch_idx, (train_input, train_label) in enumerate(train_loader):
        if args.use_cuda:
            train_input, train_label = train_input.to(DEVICE), train_label.to(DEVICE)
        optimizer.zero_grad()
        out_input, out_result, out_lambda = module(train_input)

        loss_label = criterion(train_label, out_result)
        loss_decay = criterion(train_input, out_input)
        loss = args.reg_lambda * loss_label + loss_decay
        # loss = loss_label
        loss.backward()
        optimizer.step()
        loss_train += loss.data.item()

    # scheduler.step()

    module.eval()
    loss_val = 0
    loss_val_label = 0
    loss_val_decay = 0
    for batch_idx, (val_input, val_label) in enumerate(val_loader):
        if args.use_cuda:
            val_input, val_label = val_input.to(DEVICE), val_label.to(DEVICE)
        with torch.no_grad():
            val_out, val_result, val_lambda = module(val_input)
            # val_lambda = val_lambda.to(DEVICE)
        loss_label = criterion(val_label, val_result)
        loss_decay = criterion(val_out, val_input)
        loss = args.reg_lambda * loss_label + loss_decay
        # loss = loss_label

        loss_val += loss.data.item()
        loss_val_decay += loss_decay.data.item()
        loss_val_label += loss_label.data.item()

    loss_train /= args.n_training
    loss_val /= args.n_validation
    loss_val_decay /= args.n_validation
    loss_val_label /= args.n_validation
    scheduler.step(loss_val)

    if epoch % args.test_epoch == 0 or epoch == args.n_epochs:
        for batch_idx, (test_input, test_label) in enumerate(test_loader):
            if args.use_cuda:
                test_input, test_label = test_input.to(DEVICE), test_label
            with torch.no_grad():
                test_out, test_result, test_lambda = module(test_input)
            test_result = test_result.cpu().detach().numpy()
            test_label = test_label.numpy()

            save_result_path = os.path.join(experiments_root, 'results', str(epoch))
            if not os.path.exists(save_result_path):
                os.makedirs(save_result_path)
                                            
            label_dim = test_label.shape[1]
            X = np.arange(label_dim)
            Y = np.arange(label_dim)
            X,Y = np.meshgrid(X, Y)
            test_result = test_result[0].reshape(label_dim, label_dim)
            test_label = test_label[0]

            figure = plt.figure(figsize=[10, 5])
            axes1 = figure.add_subplot(121, projection='3d')
            axes2 = figure.add_subplot(122, projection='3d')
            axes1.plot_surface(X, Y, test_result, cmap='rainbow')
            axes2.plot_surface(X, Y, test_label,cmap='rainbow')
            plt.savefig(os.path.join(save_result_path, 'data{}_mesh'.format(str(batch_idx))))

            plt.figure(figsize=[10, 5])
            plt.subplot(1, 2, 1)
            plt.contour(X, Y, test_result, 20, cmap='rainbow')
            plt.subplot(1, 2, 2)
            plt.contour(X, Y, test_label, 20, cmap='rainbow')
            plt.savefig(os.path.join(save_result_path, 'data{}_contour'.format(str(batch_idx))))


    logger.info("Epochs: %d / %d, Time: %.1f, Training loss: %.3f, Validation loss: %.3f, Validation label loss: %.3f, Validation decay loss: %.3f",
                epoch, args.n_epochs, time.time() - epoch_start_time, loss_train, loss_val, loss_val_label, loss_val_decay)

    return loss_train, loss_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--tag', type=str, default='', help='the tag for training model')
    parser.add_argument('--output_dir', type=str, default='./Experiments', help='output directory')
    parser.add_argument('--no_cuda', action='store_true', help="avoid using CUDA when available")
    parser.add_argument('--device', type=str, default="cuda:0", help="GPU device number")
    # dataset parameters
    parser.add_argument('--batch_size', type=int, default=100, help='batch size used during training')
    #module parameters
    parser.add_argument('--module_type', type=str, default='SimpleViT', help='type of the module')
    parser.add_argument('--n_layers', type=int, default=10, help='number of feed forward layers in the module')
    parser.add_argument('--n_heads', type=int, default=10, help='head number of multi-head attention in the module')
    parser.add_argument('--patch_size', type=int, default=20, help='head number of multi-head attention in the module')
    parser.add_argument('--image_size', type=int, default=100, help='head number of multi-head attention in the module')
    # training parameters
    parser.add_argument('--reg_lambda', type=float, default=0.1, help='the regularization parameter')
    parser.add_argument('--n_training', type=int, default=80000*0.9, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=80000*0.1, help='# of validation data')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate for adam optimizer used for the module')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs used to train the module')
    parser.add_argument('--save_epoch', type=int, default=5, help='saving checkpoints at the end of epochs')
    parser.add_argument('--test_epoch', type=int, default=5, help='test result at the end of epochs')
    parser.add_argument('--numpy_seed', type=int, default=100)
    parser.add_argument('--torch_seed', type=int, default=76)

    args = parser.parse_args()
    # wandb.init(config=args)
    # config = wandb.config

    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = False

    experiments_root = os.path.join(args.output_dir, '{}_{}'.format(args.module_type, util.get_timestamp()))
    if not os.path.exists(experiments_root):
        os.makedirs(experiments_root)

    file_handler = logging.FileHandler(filename=os.path.join(experiments_root, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )
    util.print_args(logger, args, experiments_root)

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed(args.torch_seed)
    torch.cuda.manual_seed_all(args.torch_seed)
    torch.backends.cudnn.deterministic = True

    train_loader, val_loader, test_loader = dataset.load_dataloader_exist(args.batch_size, args.device)

    module = set_module(args)
    # optimizer = torch.optim.Adam(module.parameters(), lr=args.lr)
    optimizer = torch.optim.RMSprop(module.parameters(), lr=args.lr, alpha=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, verbose=True, T_mult=2)
    start_epoch = 1
    criterion = torch.nn.MSELoss(reduction='sum')
    # criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(start_epoch, args.n_epochs + 1):

        if epoch < args.n_epochs:
            train_loss, val_loss = train(args=args, module=module, optimizer=optimizer, criterion=criterion,
                                           scheduler=scheduler, train_loader=train_loader, val_loader=val_loader,
                                           test_loader=test_loader, epoch=epoch, experiments_root=experiments_root)
            # metrics = {
            #     "train_loss": train_loss,
            #     "val_loss": val_loss
            # }
            # wandb.log(metrics)

        if epoch % args.save_epoch == 0 or epoch == args.n_epochs:
            util.save(module, optimizer, scheduler, args, epoch, experiments_root)
            # util.up_img(epoch)
