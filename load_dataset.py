import os
import torch
import logging
import numpy as np
from scipy.interpolate import interp2d
from torch.utils.data import Dataset

class Laplace2DDataset(Dataset):
    def __init__(self, dataroot='./Laplace2D', split='train', max_b=2, max_t=2, max_D=1, max_T=1, max_ending=0.03):
        super(Laplace2DDataset, self).__init__()
        self.split = split
        self.max_ending = max_ending
        self.max_D = max_D
        self.max_T = max_T
        self.decay_data = np.load(os.path.join(dataroot, split + '_input.npy'))
        self.label_data = np.load(os.path.join(dataroot, split + '_label.npy'))
        
        self.decay_dim = self.decay_data.shape[1]
        self.label_dim = self.label_data.shape[1]
        self.b = np.linspace(max_b / self.decay_data.shape[1], max_b, self.decay_data.shape[1])
        self.t = np.linspace(max_t / self.decay_data.shape[2], max_t, self.decay_data.shape[2])
        self.range_D = np.linspace(self.max_D / self.label_dim, self.max_D, self.label_dim)
        self.range_T = np.linspace(self.max_T / self.label_dim, self.max_T, self.label_dim)


    def __getitem__(self, index):
        decay_data = self.decay_data[index]
        label_data = self.label_data[index]
        # label_D = np.sum(label_data, axis=1) / np.max(np.sum(label_data, axis=1))
        # label_T = np.sum(label_data, axis=0) / np.max(np.sum(label_data, axis=0))
        # label_D = torch.from_numpy(label_D.copy()).float()
        # label_T = torch.from_numpy(label_T.copy()).float()

        decay_data = decay_data / np.max(decay_data)
        label_data = label_data / np.max(label_data)
        label_data = torch.from_numpy(label_data.copy()).float()

        b = self.b
        t = self.t

        max_row = np.max(decay_data, axis=1)
        max_col = np.max(decay_data, axis=0)
        row_position = np.where(max_row > self.max_ending)
        col_position = np.where(max_col > self.max_ending)
        new_b = b[row_position]
        new_t = t[col_position]

        new_t = np.linspace(np.min(new_t), np.max(new_t), self.decay_dim)
        new_b = np.linspace(np.min(new_b), np.max(new_b), self.decay_dim)

        interp_func = interp2d(t, b, decay_data, kind='cubic')
        decay_data = interp_func(new_t, new_b)
        decay_data = decay_data / decay_data[0, 0]
        decay_data = torch.from_numpy(decay_data.copy()).float()

        new_b = torch.from_numpy(new_b.copy()).float()
        new_t = torch.from_numpy(new_t.copy()).float()
        range_D = torch.from_numpy(self.range_D.copy()).float()
        range_T = torch.from_numpy(self.range_T.copy()).float()
        
        KD = torch.exp(-torch.matmul(new_b.unsqueeze(1), 1 / range_D.unsqueeze(0)))
        KT = torch.exp(-torch.matmul(new_t.unsqueeze(1), 1 / range_T.unsqueeze(0)))
        # decay_data = torch.matmul(torch.matmul(KD, label_data), KT.permute(1, 0))
        # decay_data = decay_data / decay_data[0, 0]

        concat_data = torch.concat([KD.unsqueeze(0), decay_data.unsqueeze(0), KT.unsqueeze(0)])

        return (
            {'decay_data': decay_data, 'concat_data': concat_data, 'label_data': label_data, 'b': new_b, 't': new_t,'Index': index}
        )

    def __len__(self):
        return self.decay_data.shape[0]

def create_dataset(args, phase):
    '''create dataset'''
    dataset = Laplace2DDataset(dataroot=args.data_root,
                split=phase,
                max_b=args.max_b,
                max_t=args.max_t,
                max_ending=args.max_ending
                )
    logger = logging.getLogger('base')
    logger.info('[{:s}] Dataset is created.'.format(phase))
    return dataset

def create_dataloader(dataset, args, phase):
    '''create dataloader '''
    if phase in ['train', 'valid']:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True)
    elif phase == 'test':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))