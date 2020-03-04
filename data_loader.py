import os, sys
from torchvision import transforms
import torch, torch.utils
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import bisect
import torchvision
import PIL
import struct


class HPCP(Dataset):
    def __init__(self, mode='train', out_length=None):
        self.indir = 'data/youtube_hpcp_npy/'
        if mode == 'train':
            filepath = 'hpcp/train_list_6.txt'
        elif mode == 'val':
            filepath = 'hpcp/val_list.txt'
        elif mode == 'songs350':
            self.indir = 'data/you350_hpcp_npy/'
            filepath = 'hpcp/you350_list.txt'
        elif mode == 'test':
            filepath = 'hpcp/hpcp_test_list.txt'
        elif mode == 'songs80':
            self.indir = 'data/80_hpcp_npy/'
            filepath = 'hpcp/songs80_list.txt'
        elif mode == 'songs2000':
            self.indir = 'data/songs2000_hpcp_npy/'
            filepath = 'hpcp/songs2000_list.txt'

        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.out_length = out_length

    def __getitem__(self, index):
        # data shape is [394, 23]
        filename = self.file_list[index].strip()
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        in_path = self.indir + filename + '.npy'
        data = np.load(in_path)
        # Cut to 394
        data = cut_data(data, self.out_length)
        # 12 to 23
        data = torch.from_numpy(np.concatenate((data, data[:, 0:-1]), axis=1))
        data = data.permute(1, 0).unsqueeze(0)
        return data, set_id

    def __len__(self):
        return len(self.file_list)


def cut_data(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = np.random.randint(max_offset)
            data = data[offset:(out_length + offset), :]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0, offset), (0, 0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0, offset), (0, 0)), "constant")
    return data


def cut_data_front(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            # offset = np.random.randint(max_offset)
            offset = 0
            data = data[offset:(out_length + offset), :]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0, offset), (0, 0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0, offset), (0, 0)), "constant")
    return data


def shorter(feature, mean_size=2):
    length, height = feature.shape
    new_f = np.zeros((int(length / mean_size), height), dtype=np.float64)
    for i in range(int(length / mean_size)):
        new_f[i, :] = feature[i * mean_size:(i + 1) * mean_size, :].mean(axis=0)
    return new_f


def gen_sm(fa, fb):
    l, r = fa.shape
    sm = np.ones((l, l)) * 100
    for k in range(12):
        fb = np.append(fb[:, 1:], fb[:, 0:1], axis=1)
        query = torch.from_numpy(fa).repeat(1, l).view(-1, r)
        ref = torch.from_numpy(fb).repeat(l, 1).view(-1, r)
        sm = np.min([sm, (query - ref).norm(dim=1).view(l, l).numpy()], axis=0)
        # sm = np.min([sm,(1-np.matmul(fa,fb.T))/2],axis=0)
    max_dis = np.max(sm)
    sm = (- sm + max_dis) / max_dis
    return sm


class SM_HPCP(Dataset):
    def __init__(self):
        self.indir = 'data/youtube_hpcp_npy/'
        filepath = 'hpcp/triplet_hpcp_list.txt'
        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]

    def __getitem__(self, index):
        filename = self.file_list[index].strip()
        name_list = filename.split(',')
        data_list = []
        for name in name_list:
            data = np.load(self.indir + name + '.npy')
            data = shorter(cut_data(data, 360))
            # data = torch.from_numpy(np.concatenate((data, data[:,0:-1]), axis=1))
            # data = data.permute(1,0).unsqueeze(0)
            data_list.append(data)
        sm1 = torch.from_numpy(gen_sm(data_list[0], data_list[1])).unsqueeze(0)
        sm2 = torch.from_numpy(gen_sm(data_list[0], data_list[2])).unsqueeze(0)
        return sm1, sm2, 0, 1

    def __len__(self):
        return len(self.file_list)


class triplet_HPCP(Dataset):
    def __init__(self, out_length):
        self.indir = 'data/youtube_hpcp_npy/'
        filepath = 'hpcp/triplet_hpcp_list.txt'
        self.out_length = out_length
        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]

    def __getitem__(self, index):
        filename = self.file_list[index].strip()
        name_list = filename.split(',')
        data_list = []
        for name in name_list:
            data = np.load(self.indir + name + '.npy')
            data = cut_data(data, self.out_length)
            data = torch.from_numpy(np.concatenate((data, data[:, 0:-1]), axis=1))
            data = data.permute(1, 0).unsqueeze(0)
            data_list.append(data)
        return data_list[0], data_list[1], data_list[2]

    def __len__(self):
        return len(self.file_list)


class triplet_CQT(Dataset):
    def __init__(self, out_length):
        self.indir = 'data/youtube_cqt_npy/'
        filepath = 'hpcp/triplet_hpcp_list.txt'
        self.out_length = out_length
        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]

    def __getitem__(self, index):
        transform_train = transforms.Compose([
            lambda x: x.T,
            lambda x: change_speed(x, 0.7, 1.3),
            # lambda x : x-np.mean(x),
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: cut_data(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.permute(1, 0).unsqueeze(0),
        ])
        filename = self.file_list[index].strip()
        name_list = filename.split(',')
        data_list = []
        for name in name_list:
            data = np.load(self.indir + name + '.npy')
            data = transform_train(data)
            data_list.append(data)
        return data_list[0], data_list[1], data_list[2]

    def __len__(self):
        return len(self.file_list)


class CENS(Dataset):
    def __init__(self, mode='train', out_length=None):
        self.indir = 'data/youtube_cens_npy/'
        if mode == 'train':
            filepath = 'hpcp/hpcp_train_list.txt'
        elif mode == 'val':
            filepath = 'hpcp/val_list.txt'
        elif mode == 'songs350':
            self.indir = 'data/you350_cens_npy/'
            filepath = 'hpcp/you350_list.txt'
        elif mode == 'test':
            filepath = 'hpcp/hpcp_test_list.txt'
        elif mode == 'songs80':
            self.indir = 'data/covers80_cens_npy/'
            filepath = 'hpcp/songs80_list.txt'
        # elif mode == 'songs2000':
        #    self.indir = 'data/songs2000_hpcp_npy/'
        #    filepath = 'hpcp/songs2000_list.txt'

        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.out_length = out_length

    def __getitem__(self, index):

        filename = self.file_list[index].strip()
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        in_path = self.indir + filename + '.npy'
        data = np.load(in_path)  # from 12xN to Nx12
        data = data.T
        # Cut to 394
        data = cut_data(data, self.out_length)
        # 12 to 23
        data = torch.from_numpy(np.concatenate((data, data[:, 0:-1]), axis=1)).float()
        data = data.permute(1, 0).unsqueeze(0)
        return data, set_id

    def __len__(self):
        return len(self.file_list)


class CQT(Dataset):
    def __init__(self, mode='train', out_length=None):
        # self.indir = 'data/youtube_cqt_npy/'
        self.indir = 'data/youtube_cqt_npy/'
        self.mode = mode
        if mode == 'train':
            # filepath='hpcp/hpcp_train_list.txt'
            filepath = 'hpcp/SHS100K-TRAIN_6'
            # filepath='hpcp/train_list_6.txt'
            # self.new_map=np.load('hpcp/new_map.npy')
        elif mode == 'val':
            # filepath='hpcp/val_list.txt'
            filepath = 'hpcp/SHS100K-VAL'
        elif mode == 'songs350':
            self.indir = 'data/you350_cqt_npy/'
            filepath = 'hpcp/you350_list.txt'
        elif mode == 'test':
            filepath = 'hpcp/SHS100K-TEST'
            # filepath='hpcp/test_list.txt'
        elif mode == 'songs80':
            self.indir = 'data/covers80_cqt_npy/'
            filepath = 'hpcp/songs80_list.txt'
        elif mode == 'songs2000':
            self.indir = 'data/songs2000_cqt_npy/'
            filepath = 'hpcp/songs2000_list.txt'
        elif mode == 'new80':
            self.indir = 'data/songs2000_cqt_npy/'
            filepath = 'hpcp/new80_list.txt'
        elif mode == 'Mazurkas':
            self.indir = 'data/Mazurkas_cqt_npy/'
            filepath = 'hpcp/Mazurkas_list.txt'
        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.out_length = out_length

    def __getitem__(self, index):
        transform_train = transforms.Compose([
            lambda x: x.T,
            lambda x: change_speed(x, 0.7, 1.3),
            # lambda x : x-np.mean(x),
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: cut_data(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.permute(1, 0).unsqueeze(0),
        ])
        transform_test = transforms.Compose([
            lambda x: x.T,
            # lambda x : x-np.mean(x),
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: cut_data_front(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.permute(1, 0).unsqueeze(0),
        ])
        filename = self.file_list[index].strip()
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        in_path = self.indir + filename + '.npy'
        data = np.load(in_path)  # from 12xN to Nx12

        if self.mode is 'train':
            data = transform_train(data)
        else:
            data = transform_test(data)
        return data, int(set_id), int(version_id)

    def __len__(self):
        return len(self.file_list)


class CQT_Yang(Dataset):
    def __init__(self, mode='train', out_length=None):
        # self.indir = 'data/youtube_cqt_npy/'
        self.indir = '/S1/DAA/yzs/cqt_youtube126322/'
        self.mode = mode
        if mode == 'train':
            filepath = 'hpcp/SHS100K-TRAIN_6_Yang'
        elif mode == 'val':
            # filepath='hpcp/val_list.txt'
            filepath = 'hpcp/SHS100K-VAL_Yang'
        elif mode == 'test':
            filepath = 'hpcp/SHS100K-TEST_Yang'
            # filepath='hpcp/test_list.txt'
        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.out_length = out_length

    def __getitem__(self, index):
        transform_train = transforms.Compose([
            lambda x: change_speed(x, 0.7, 1.3),
            # lambda x : x-np.mean(x),
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: cut_data(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.permute(1, 0).unsqueeze(0),
        ])
        transform_test = transforms.Compose([
            # lambda x : x-np.mean(x),
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: cut_data_front(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.permute(1, 0).unsqueeze(0),
        ])
        filename = self.file_list[index].strip()
        set_id, version_id = filename.split('.')[0].split('_')
        in_path = self.indir + set_id + '/' + filename + '.mp3.cqt'
        set_id, version_id = int(set_id), int(version_id)

        # data = np.load(in_path) # from 12xN to Nx12
        data = None
        with open(in_path, 'rb') as f:
            content = f.read()
            l, _ = struct.unpack("ii", content[:8])
            data = np.array(struct.unpack("d" * (84 * l), content[8:])).reshape((-1, 84))
        if self.mode is 'train':
            data = transform_train(data)
        else:
            data = transform_test(data)
        return data, int(set_id)

    def __len__(self):
        return len(self.file_list)


def change_speed(data, l=0.7, r=1.5):  # change data.shape[0]
    new_len = int(data.shape[0] * np.random.uniform(l, r))
    maxx = np.max(data) + 1
    data0 = PIL.Image.fromarray((data * 255.0 / maxx).astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize(size=(new_len, data.shape[1])),
    ])
    new_data = transform(data0)
    return np.array(new_data) / 255.0 * maxx


class get_triloader(Dataset):
    def __init__(self, out_length=None):
        self.out_length = out_length
        self.indir = 'data/youtube_cqt_npy/'
        filepath = 'hpcp/SHS100K-TRAIN_6'
        self.file_list = []
        self.dic = {}
        with open(filepath, 'r') as fp:
            for line in fp:
                self.file_list.append(line.rstrip())
                set_id, version_id = line.rstrip().split('_')
                set_id, version_id = int(set_id), int(version_id)
                if set_id not in self.dic:
                    self.dic[set_id] = []
                self.dic[set_id].append(version_id)

    def __getitem__(self, index):
        filename = self.file_list[index].strip()
        set_id, version_id = filename.split('.')[0].split('_')
        set_id, version_id = int(set_id), int(version_id)
        in_path = self.indir + filename + '.npy'
        data = np.load(in_path).T
        data = cut_data(data, self.out_length)  # L, 84
        data = torch.from_numpy(data).float()
        data = data.permute(1, 0).unsqueeze(0)
        data0 = data

        filename1, filename2 = generate_triplet(self.dic, set_id, version_id)

        set_id1 = set_id
        in_path = self.indir + filename1 + '.npy'
        data = np.load(in_path).T
        data = cut_data(data, self.out_length)  # L, 84
        data = torch.from_numpy(data).float()
        data = data.permute(1, 0).unsqueeze(0)
        data1 = data

        set_id2 = int(filename2.split('_')[0])
        in_path = self.indir + filename2 + '.npy'
        data = np.load(in_path).T
        data = cut_data(data, self.out_length)  # L, 84
        data = torch.from_numpy(data).float()
        data = data.permute(1, 0).unsqueeze(0)
        data2 = data

        return data0, int(set_id), data1, int(set_id1), data2, int(set_id2)

    def __len__(self):
        return len(self.file_list)


import random


def generate_triplet(dic, set_id, ver_id):
    # 给当前的歌曲信息，构造一个三元组
    # positive
    p_set = set_id
    p_ver2 = random.sample(dic[set_id], 2)
    if p_ver2[0] == ver_id:
        p_ver = p_ver2[1]
    else:
        p_ver = p_ver2[0]
    # negative
    n_set2 = random.sample(dic.keys(), 2)
    if n_set2[0] == set_id:
        n_set = n_set2[1]
    else:
        n_set = n_set2[0]
    n_ver = random.sample(dic[n_set], 1)[0]
    return str(p_set) + '_' + str(p_ver), str(n_set) + '_' + str(n_ver)


if __name__ == '__main__':
    train_dataset = HPCP('train', 394)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)
