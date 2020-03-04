import torch
from torch.utils.data import DataLoader
from processing import *
from utility import *
from data_loader import *
import models


def get_norm_feature(model, cqt):
    my_transform = transforms.Compose([
        lambda x: x.T,
        lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
        lambda x: cut_data_front(x, 400),
        lambda x: torch.Tensor(x),
        lambda x: x.permute(1, 0).unsqueeze(0),
    ])

    cqt = my_transform(cqt).unsqueeze(0)
    myfeature = norm(model.get_fixed_out(cqt).data.cpu().numpy())
    return myfeature


def cal_similar(music_list):
    music1 = music_list[0]
    music2 = music_list[1]
    print(music_list)
    path1 = './sim1.' + music1.filename.split('.')[1]
    path2 = './sim2.' + music2.filename.split('.')[1]
    music1.save(path1)
    music2.save(path2)

    model = getattr(models, 'SMCNN_4')()
    model.load("1227_08_47_43.pth")

    cqt1 = get_cqt(path1)[1]
    cqt2 = get_cqt(path2)[1]
    ft1 = get_norm_feature(model, cqt1)
    ft2 = get_norm_feature(model, cqt2)

    ft2 = ft2.T
    dis2d = np.matmul(ft1, ft2)

    return str(round(dis2d[0][0] * 100, 2))


def get_similar(cqt, recal):
    model = getattr(models, 'SMCNN_4')()
    model.load("1227_08_47_43.pth")
    dataloader = DataLoader(CQT('songs80', out_length=400), 1, shuffle=False, num_workers=1)
    features, sets, versions = None, None, None

    myfeature = get_norm_feature(model, cqt)

    if recal == 1:
        for ii, (data, set_id, version_id) in enumerate(dataloader):
            input = data.to('cpu')
            feature = model.get_fixed_out(input)
            feature = feature.data.cpu().numpy()
            if features is not None:
                features = np.concatenate((features, feature), axis=0)
                versions = np.concatenate((versions, version_id), axis=0)
                sets = np.concatenate((sets, set_id), axis=0)
            else:
                features = feature
                versions = version_id
                sets = set_id
        np.save('features.npy', features)
        np.save('versions.npy', versions)
        np.save('sets.npy', sets)

    else:
        features = np.load('features.npy')
        versions = np.load('versions.npy')
        sets = np.load('sets.npy')

    features = norm(features)

    features = features.T
    dis2d = -np.matmul(myfeature, features)

    sort_ind = np.argsort(dis2d)
    target_ind = sort_ind[0][:10].tolist()
    target_pos = []
    for ind in target_ind:
        print(dis2d[0][ind])
        target_pos.append((sets[ind], versions[ind]))

    info = []
    i = 0
    for s, v in target_pos:
        infos = get_name_by_ind((s, v))
        info.append(dict())
        info[i]['artist'] = infos[0]
        info[i]['album'] = infos[1]
        info[i]['title'] = infos[2]
        info[i]['score'] = str(round(-dis2d[0][target_ind[i]] * 100, 2))
        info[i]['format'] = infos[3]
        info[i]['set'] = str(s)
        info[i]['version'] = str(v)
        shutil.copyfile(get_path_by_ind((s, v)), './static/result/songs' + str(s) + '_' + str(v) + '.' + infos[3])
        i = i + 1

    print(info)

    return info



