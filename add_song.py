from processing import *
from similar import *
import numpy as np

def add_songs(musics):
    song_path = './data/covers80/coversongs/covers32k'
    dirs = os.listdir(song_path)
    dirs.sort(key=lambda x: int(x.split('-')[0]))
    npy_path = './data/covers80_cqt_npy'
    error_list = []
    model = getattr(models, 'SMCNN_4')()
    model.load("1227_08_47_43.pth")

    for music in musics:
        music.filename = music.filename.replace('~', '+')
        path = './tmp.' + music.filename.split('.')[1]
        music.save(path)
        infos = music.filename.split('+')

        if len(infos) != 3:
            error_list.append(music.filename)
            continue

        title = infos[2].split('.')[0]
        set_id = judge_exist_set(title)

        need_update = False
        version_id = 0

        if set_id == -1:
            need_update = True
            set_id = len(dirs)
            set_path = song_path + '/' + str(set_id) + '-' + title
            os.mkdir(set_path)
            version_id = 0
            save_path = set_path + '/' + str(version_id) + '-' + music.filename
            shutil.copyfile(path, save_path)

        else:
            set_path = song_path + '/' + dirs[set_id]
            exist = judge_exist_version(set_path, music.filename)
            if exist != 0:
                need_update = True
                version_id = exist
                save_path = set_path + '/' + str(version_id) + '-' + music.filename
                shutil.copyfile(path, save_path)

        if need_update:
            cqt = get_cqt(path)[1]
            np.save(npy_path + '/' + str(set_id) + '_' + str(version_id) + '.npy', cqt)
            feature = get_norm_feature(model, cqt)
            features = np.load('features.npy')
            versions = np.load('versions.npy')
            sets = np.load('sets.npy')

            version_list = []
            set_list = []
            version_list.append(version_id)
            set_list.append(set_id)
            features = np.concatenate((features, feature), axis=0)
            versions = np.concatenate((versions, np.array(version_list)), axis=0)
            sets = np.concatenate((sets, np.array(set_list)), axis=0)

            np.save('features.npy', features)
            np.save('versions.npy', versions)
            np.save('sets.npy', sets)

    return error_list

