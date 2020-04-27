import os
import numpy as np
import sys
path = "./data/youtube_cqt_npy/"
train_path = "./data/youtube_train_"
test_path = "./data/youtube_test_"
val_path = "./data/youtube_val_"


def gen_crossval(scale=1):

    mscs = os.listdir(path)
    ids = []
    dic = {}
    for msc in mscs:
        set_id = msc.split('_')[0]
        version_id = msc.split('_')[1].split('.')[0]
        if set_id not in dic:
            dic[set_id] = []
            dic[set_id].append(version_id)
        else:
            dic[set_id].append(version_id)

    for key in dic:
        if len(dic[key]) > 1:
            ids.append(key)

    ids = np.array(ids)
    lens = int(len(ids) * scale)
    print(lens)
    train_len = int(lens * 4 / 5)
    print(train_len)
    test_len = int((lens - train_len) / 2)

    np.random.seed(int(25 * scale))
    np.random.shuffle(ids)

    with open(train_path + str(scale) + '.txt', 'w') as f:
        for i in range(0, train_len):
            print(i)
            for msc in mscs:
                if ids[i] == msc.split('_')[0]:
                    f.write(msc.split('.')[0] + '\n')
        print('train generated!!\n')
    
    with open(test_path + str(scale) + '.txt', 'w') as f:
        for i in range(train_len, train_len + test_len):
            for msc in mscs:
                if ids[i] == msc.split('_')[0]:
                    f.write(msc.split('.')[0] + '\n')
        print('test generated!!\n')

    with open(val_path + str(scale) + '.txt', 'w') as f:
        for i in range(train_len + test_len, lens):
            for msc in mscs:
                if ids[i] == msc.split('_')[0]:
                    f.write(msc.split('.')[0] + '\n')
        print('validate generated!!\n')



if __name__ == '__main__':
    sscale = sys.argv[1]
    print(sscale)
    gen_crossval(float(sscale))
    
    

