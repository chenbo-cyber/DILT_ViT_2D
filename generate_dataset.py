import os
import torch
import numpy as np
import scipy.io as scio
import argparse
from tqdm import trange
from sklearn.model_selection import train_test_split as train_val

def gaussian_blind_noise(S, dB):
    """
    Add Gaussian noise to the input signal. The std of the gaussian noise is uniformly chosen between 0 and 1/sqrt(snr).
    """
    snr = np.exp(np.log(10) * float(dB) / 10)
    num_samples, signal_dim, num_fre = np.shape(S)
    noise_S = np.zeros([num_samples, signal_dim, num_fre])
    # sigma_max = np.sqrt(1. / snr)
    # sigmas = sigma_max * np.random.rand(num_samples)
    sigma = np.sqrt(1. / snr)

    for i in np.arange(num_samples):
        noise = np.random.randn(signal_dim, num_fre) * S[i, :, :]
        # mult = sigmas[i] * np.linalg.norm(s, 2) / (np.linalg.norm(noise, 2))
        mult = sigma * np.linalg.norm(S[i, :, :], ord='fro') / (np.linalg.norm(noise, ord='fro'))
        noise = noise * mult
        
        noise_S[i, :, :] = S[i, :, :] + noise
    return noise_S

def Gaussian_distribution_log(max_D, floor_D, avg, num, sig):
    xgrid = np.logspace(np.log10(floor_D), np.log10(max_D), num)
    result = np.exp(-(np.log10(xgrid) - avg.T)**2 / (2 * sig**2)) / (xgrid * sig * np.sqrt(2 * np.pi))
    result[np.isnan(result)] = 0
    return result/np.max(result, axis=-1)[:, np.newaxis]

def generate_dataset(args):
    input_signal, label = gen_signal(args=args)
    
    train_input, val_input, train_label, val_label = train_val(input_signal, label, test_size=args.ratio, random_state=42)

    train_input = torch.from_numpy(train_input).float()
    val_input = torch.from_numpy(val_input).float()
    train_label = torch.from_numpy(train_label).float()
    val_label = torch.from_numpy(val_label).float()

    np.save(os.path.join(args.output_dir, "train_input"), train_input)
    np.save(os.path.join(args.output_dir, "valid_input"), val_input)
    np.save(os.path.join(args.output_dir, "train_label"), train_label)
    np.save(os.path.join(args.output_dir, "valid_label"), val_label)

    print('##############################finished##############################')

def load_dataloader_exist(batch_size, device):
    DEVICE = torch.device("cpu")

    train_input = np.load("./Dataset/train_input.npy")
    train_label = np.load("./Dataset/train_label.npy")
    val_input = np.load("./Dataset/val_input.npy")
    val_label = np.load("./Dataset/val_label.npy")
    test_input = np.load("./Dataset/test_input.npy")
    test_label = np.load("./Dataset/test_label.npy")

    train_input = torch.from_numpy(train_input).float().to(DEVICE)
    val_input = torch.from_numpy(val_input).float().to(DEVICE)
    test_input = torch.from_numpy(test_input).float().to(DEVICE)
    train_label = torch.from_numpy(train_label).float().to(DEVICE)
    val_label = torch.from_numpy(val_label).float().to(DEVICE)
    test_label = torch.from_numpy(test_label).float().to(DEVICE)

    train_dataset = torch.utils.data.TensorDataset(train_input, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_input, val_label)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=2, shuffle=False)

    test_dataset = torch.utils.data.TensorDataset(test_input, test_label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)

    return train_loader, val_loader, test_loader

# to generate gaussian distribution
def Gaussian_distribution(max_D, floor_D, avg, num, sig):
    avg = avg.T
    xgrid = np.linspace(floor_D, max_D, num)
    sqrt_2pi=np.power(2*np.pi,0.5)
    coef=1/(sqrt_2pi*sig)
    powercoef=-1/(2*np.power(sig,2))
    mypow=powercoef*(np.power((xgrid-avg),2))
    result = coef*(np.exp(mypow))
    result[np.isnan(result)] = 0
    return result/np.max(result)

def gen_signal(args):
    S = np.zeros([args.num_samples, args.input_dim, args.input_dim])
    label = np.zeros([args.num_samples, args.label_dim, args.label_dim])
    b = np.linspace(args.max_b / args.input_dim, args.max_b, args.input_dim)
    t = np.linspace(args.max_t / args.input_dim, args.max_t, args.input_dim)
    # b = scio.loadmat('Dataset/T2-T2_mt50ms.mat')['t2values']
    # t = scio.loadmat('Dataset/T2-T2_mt50ms.mat')['t2values']
    if args.axis_type == 0:
        tt = np.linspace(args.max_T2 / args.label_dim, args.max_T2, args.label_dim)[np.newaxis, :]
        dd = np.linspace(args.max_D / args.label_dim, args.max_D, args.label_dim)[np.newaxis, :]
    elif args.axis_type == 1:
        tt = np.logspace(np.log10(args.floor_T2), np.log10(args.max_T2), args.label_dim)[np.newaxis, :]
        dd = np.logspace(np.log10(args.floor_D), np.log10(args.max_D), args.label_dim)[np.newaxis, :]
    KD = np.exp(-np.dot(b[:, np.newaxis], 1 / dd))
    KT = np.exp(-np.dot(t[:, np.newaxis], 1 / tt))
    # KD = np.exp(-np.dot(b, 1 / dd))
    # KT = np.exp(-np.dot(t, 1 / tt))
    nDT = np.random.randint(1, args.num_component + 1, args.num_samples)

    for i in trange(args.num_samples):
        D = np.ones([1, args.num_component]).astype(float)
        T2 = np.zeros([1, args.num_component]).astype(float)
        signal = np.zeros([args.input_dim, args.input_dim]).astype(float)
        label1 = np.zeros([args.label_dim, args.label_dim]).astype(float)

        if args.axis_type == 1:
            for j in np.arange(nDT[i]):
                D_value = np.random.uniform(low=np.log10(args.floor_D), high=np.log10(args.max_D))
                condition = True
                while condition:
                    D_value = np.random.uniform(low=np.log10(args.floor_D), high=np.log10(args.max_D))
                    condition = np.min(np.abs(D - D_value)) < args.min_sep_D or D_value < np.log10(args.floor_D)
                D[0, j] = D_value

                T2_value = np.random.uniform(low=np.log10(args.floor_T2), high=np.log10(args.max_T2))
                condition = True
                while condition:
                    T2_value = np.random.uniform(low=np.log10(args.floor_T2), high=np.log10(args.max_T2))
                    condition = np.min(np.abs(T2 - T2_value)) < args.min_sep_T2 or T2_value < np.log10(args.floor_T2)
                T2[0, j] = T2_value
            label_D = Gaussian_distribution_log(args.max_D, args.floor_D, D, args.label_dim, sig=args.sig_D)
            label_T2 = Gaussian_distribution_log(args.max_T2, args.floor_T2, T2, args.label_dim, sig=args.sig_T2)

        elif args.axis_type == 0:
            for j in np.arange(nDT[i]):
                D_value = np.random.uniform(low=args.floor_D, high=args.max_D)
                condition = True
                while condition:
                    D_value = np.random.uniform(low=args.floor_D, high=args.max_D)
                    condition = np.min(np.abs(D - D_value)) < args.min_sep_D or D_value < args.floor_D
                D[0, j] = D_value

                T2_value = np.random.uniform(low=args.floor_T2, high=args.max_T2)
                condition = True
                while condition:
                    T2_value = np.random.uniform(low=args.floor_T2, high=args.max_T2)
                    condition = np.min(np.abs(T2 - T2_value)) < args.min_sep_T2 or T2_value < args.floor_T2
                T2[0, j] = T2_value
            label_D = Gaussian_distribution(args.max_D, args.floor_D, D, args.label_dim, sig=args.sig_D)
            label_T2 = Gaussian_distribution(args.max_T2, args.floor_T2, T2, args.label_dim, sig=args.sig_T2)

        for j in np.arange(nDT[i]):
            amp = max(np.abs(np.random.rand()), args.floor_amp)
            # signal = signal + amp * np.dot(np.exp(-b[:, np.newaxis] * R1[0, j]), np.exp(-t[:, np.newaxis] * R[0, j]).T)
            label1 = label1 + amp * np.dot(label_D[j, :][:, np.newaxis], label_T2[j, :][np.newaxis, :])
        signal = np.matmul(np.matmul(KD, label1), KT.T)
        S[i] = signal / np.max(signal)
        label[i] = label1 / np.max(label1)

    SN = gaussian_blind_noise(S, args.dB)

    return SN.astype('float32'), label.astype('float32')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='Dataset/', help='the path of output file')
    parser.add_argument('--num_samples', type=int, default=80000, help='the number of simulated data')
    parser.add_argument('--ratio', type=float, default=0.1, help='the ratio between training and validation')
    parser.add_argument('--floor_amp', type=float, default=0.2, help='the floor amplitude of one component')
    parser.add_argument('--input_dim', type=int, default=100, help='the dimension size of input')
    parser.add_argument('--label_dim', type=int, default=100, help='the dimension size of label')
    parser.add_argument('--num_component', type=int, default=5, help='the number of component')
    parser.add_argument('--dB', type=int, default=48, help='the SNR of addition noise')
    parser.add_argument('--axis_type', type=int, default=0, help='0: linspace, 1: logspace')
    # parameters for dimension 1
    parser.add_argument('--max_D', type=float, default=1, help='the max value of diffusion coefficient')
    parser.add_argument('--floor_D', type=float, default=0, help='the floor value of diffusion coefficient')
    parser.add_argument('--min_sep_D', type=float, default=0.1, help='the min sep between two component')
    parser.add_argument('--sig_D', type=float, default=0.02, help='the width of peaks')
    parser.add_argument('--max_b', type=float, default=5, help='the max value of b array')
    # parameters for dimension 2
    parser.add_argument('--max_T2', type=float, default=1, help='the max value of T2')
    parser.add_argument('--floor_T2', type=float, default=0, help='the floor value of T2')
    parser.add_argument('--min_sep_T2', type=float, default=0.1, help='the min sep between two component')
    parser.add_argument('--sig_T2', type=float, default=0.02, help='the width of peaks')
    parser.add_argument('--max_t', type=float, default=5, help='the max value of t array')
    args = parser.parse_args()

    generate_dataset(args=args)