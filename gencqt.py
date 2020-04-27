import librosa
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm

in_dir = '/media/data/data_share/new_youtube/crawl_data/data/'
out_dir = './data/youtube_cqt_npy/'


def CQT(args):
    in_path, out_path = args
    data, sr = librosa.load(in_path)
    if len(data)<1000:
        return
    cqt = np.abs(librosa.cqt(y=data, sr=sr))
    mean_size = 20
    height, length = cqt.shape
    new_cqt = np.zeros((height,int(length/mean_size)),dtype=np.float64)
    for i in range(int(length/mean_size)):
        new_cqt[:,i] = cqt[:,i*mean_size:(i+1)*mean_size].mean(axis=1)
    np.save(out_path, new_cqt)
    print(out_path + " saved")


print(os.cpu_count())
params =[]
for ii, (root, dirs, files) in tqdm(enumerate(os.walk(in_dir))):  
    if len(files):
        for file in files:
            in_path = os.path.join(root,file)
            set_id = root.split('/')[-1]
            out_path = out_dir + set_id + '_' + file.split('.')[0] + '.npy'
            params.append((in_path, out_path))

print('begin')
pool = Pool(os.cpu_count())
pool.map(CQT, params)
pool.close()
pool.join()


