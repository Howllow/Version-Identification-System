import librosa
import numpy as np
import os
import torch
import shutil


def get_cqt(path):
    data, sr = librosa.load(path)
    cqt = np.abs(librosa.cqt(y=data, sr=sr))
    mean_size = 20
    height, length = cqt.shape
    new_cqt = np.zeros((height, int(length / mean_size)), dtype=np.float64)
    for i in range(int(length / mean_size)):
        new_cqt[:, i] = cqt[:, i * mean_size:(i + 1) * mean_size].mean(axis=1)

    return 0, new_cqt


def rename():
    path = "./data/covers80/coversongs/covers32k"
    dirs = os.listdir(path)
    dirs.sort()
    for i in range(0, len(dirs)):
        old_dir = dirs[i]
        dir_path = path + "/" + old_dir
        print(dir_path)
        files = os.listdir(dir_path)
        files.sort()
        for j in range(0, len(files)):
            old_file = files[j]
            parts = old_file.split('+')
            new_name = old_file
            if len(parts[2].split('-')) == 2:
                new_name = parts[0] + '+' + parts[1] + '+' + parts[2].split('-')[1]
            new_file = dir_path + "/" + str(j) + "-" + new_name
            file_path = dir_path + "/" + old_file
            os.rename(file_path, new_file)
        new_dir = path + "/" + str(i) + "-" + old_dir
        os.rename(dir_path, new_dir)
    return


def get_path_by_ind(inds):
    setid = inds[0]
    verid = inds[1]
    path = "./data/covers80/coversongs/covers32k"
    dirs = os.listdir(path)
    dirs.sort(key=lambda x: int(x.split('-')[0]))
    dir_path = path + "/" + dirs[setid]
    files = os.listdir(dir_path)
    files.sort(key=lambda x: int(x.split('-')[0]))
    file_name = files[verid]
    return dir_path + "/" + file_name


def get_name_by_ind(inds):
    file_path = get_path_by_ind(inds)
    layers = file_path.split('/')
    file_name = layers[len(layers) - 1]
    names = file_name.split('+')
    authors = names[0].split('-')[1]
    album = names[1]
    title = names[2].split('.')[0]
    fformat = names[2].split('.')[1]

    return authors, album, title, fformat


def generate_song_index():
    path = "./data/covers80/coversongs/covers32k"
    dirs = os.listdir(path)
    dirs.sort(key=lambda x: int(x.split('-')[0]))
    with open('./song_id.txt', 'w') as f:
        for i in range(0, len(dirs)):
            print(dirs[i])
            f.write(str(i) + '-')
            f.write(dirs[i].split('-')[1] + '\n')


def judge_exist_set(name):
    path = "./data/covers80/coversongs/covers32k"
    dirs = os.listdir(path)
    dirs.sort(key=lambda x: int(x.split('-')[0]))
    for i in range(0, len(dirs)):
        if name == dirs[i].split('-')[1]:
            print("set exists")
            return i
    return -1


def judge_exist_version(path, filename):
    files = os.listdir(path)
    files.sort(key=lambda x: int(x.split('-')[0]))
    for file in files:
        if filename == file.split('-')[1]:
            print("version exists")
            return 0

    return len(files)


#if __name__ == '__main__':





