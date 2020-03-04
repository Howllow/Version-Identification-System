import os
import torch
from data_loader import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models
from config import opt
from torchnet import meter
from tqdm import tqdm
import numpy as np
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from utility import *
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import triplet
torch.backends.cudnn.benchmark = True
"python main.py train --model=CQT_inception --GPU=0 --batch_size=50 -- max_epoch=100 --lr=0.001"
def train(**kwargs):

    parallel = False

    #多尺度图片训练 396+
    #opt.feature, opt.model, opt.notes = 'cens', 'SPPNet','cens'
    opt._parse(kwargs)
    GPU = opt.GPU
    opt.notes = '400_train'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    opt.load_latest=False
    #opt.load_model_path = 'check_points/<class \'models.ATTSPPNet.ATTSPPNetV1\'>V1/1115_21:41:19.pth'
    data_length = opt.data_length
    # step1: configure model
    model = getattr(models, opt.model)() #从models文件夹里选出名字叫opt.model的模型
    if parallel is True:
        print("aa")
        model = torch.nn.DataParallel(model)
    if parallel is True:
        if opt.load_latest is True:
            model.module.load_latest(opt.notes)
        elif opt.load_model_path:
            model.module.load(opt.load_model_path)
    else:
        if opt.load_latest is True:
            model.load_latest(opt.notes)
        elif opt.load_model_path:
            model.load(opt.load_model_path)
    model.to(opt.device)



    # step2: data

    train_data0 = CQT('train', out_length=data_length)
    val_data350 = CQT('songs350', out_length=data_length)
    val_data80 = CQT('songs80', out_length=data_length)
    # val_data = CQT('val', out_length=data_length)
    # test_data = CQT('test', out_length=data_length)
    train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True,num_workers=10)
    val_dataloader350 = DataLoader(val_data350, 1, shuffle=False,num_workers=1)
    # val_dataloader = DataLoader(val_data, 1, shuffle=False,num_workers=1)
    val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
    # test_dataloader = DataLoader(test_data, 1, shuffle=False,num_workers=1)
    #val_dataloader2000 = DataLoader(val_data2000, 1, shuffle=False, num_workers=1)
    #step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    if parallel is True:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min',factor=opt.lr_decay,patience=1, verbose=True,min_lr=5e-6)
    #train
    best_MAP=0
    for epoch in range(opt.max_epoch):

        running_loss = 0
        num = 0
        for (data0, label0) in tqdm(train_dataloader0):
            data=data0
            label=label0
            # train model
            input = data.requires_grad_()
            input = input.to(opt.device)
            target = label.to(opt.device)
            optimizer.zero_grad()
            score, _ = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num += target.shape[0]
        running_loss /= num
        print(running_loss)
        # update learning rate
        scheduler.step(running_loss)
        # validate
        MAP=0
        print("-------------Youtube  Result---------------")
        MAP += val_slow(model, val_dataloader350, epoch)
        print("-------------Covers80 Result---------------")
        MAP += val_slow(model, val_dataloader80, epoch)
        # val_quick(model,val_dataloader)
        # val_quick(model,test_dataloader)
        if MAP>best_MAP:
            best_MAP=MAP
            print('*****************BEST*****************')
            if parallel is True:
                model.module.save(opt.notes)
            else:
                model.save(opt.notes)
        print('')
        model.train()



def multiloss_train(**kwargs):

    parallel = False
    margin = 0.9
    # 多尺度图片训练 396+
    # opt.feature, opt.model, opt.notes = 'cens', 'SPPNet','cens'
    opt._parse(kwargs)
    GPU = opt.GPU
    opt.notes = '400_train'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    opt.load_latest = False
    # opt.load_model_path = 'check_points/<class \'models.ATTSPPNet.ATTSPPNetV1\'>V1/1115_21:41:19.pth'
    data_length = opt.data_length
    test_data_length = opt.test_data_length
    # step1: configure model
    model = getattr(models, opt.model)()  # 从models文件夹里选出名字叫opt.model的模型
    if parallel is True:
        print("aa")
        model = torch.nn.DataParallel(model)
    if parallel is True:
        if opt.load_latest is True:
            model.module.load_latest(opt.notes)
        elif opt.load_model_path:
            model.module.load(opt.load_model_path)
    else:
        if opt.load_latest is True:
            model.load_latest(opt.notes)
        elif opt.load_model_path:
            model.load(opt.load_model_path)
    model.to(opt.device)
    # step1: configure model
    # step2: data

    train_data0 = get_triloader(out_length=400)
        # train_data1 = get_triloader(out_length=300)
        # train_data2 = get_triloader(out_length=400)
    val_data350 = CQT('songs350', out_length=400)
    val_data80 = CQT('songs80', out_length=400)
        # val_data = CQT('val', out_length=None)
        # test_data = CQT('test', out_length=None)
    train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # train_dataloader1 = DataLoader(train_data1, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # train_dataloader2 = DataLoader(train_data2, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=1)
    # test_dataloader = DataLoader(test_data, 1, shuffle=False, num_workers=1)
    val_dataloader80 = DataLoader(val_data80, 100, shuffle=False, num_workers=1)
    val_dataloader350 = DataLoader(val_data350, 100, shuffle=False, num_workers=1)
    # val_dataloader2000 = DataLoader(val_data2000, 1, shuffle=False, num_workers=1)
    # step3: criterion and optimizer
    criterion0 = torch.nn.TripletMarginLoss(margin=margin)
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.BCELoss()
    lr = opt.lr
    if parallel is True:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=opt.lr_decay, patience=2, verbose=True, min_lr=5e-6)
    # train
    best_MAP = 0
    best_MAP1 = 0

    for epoch in range(opt.max_epoch):
        running_loss, running_loss0, running_loss1,running_loss2 = 0, 0, 0, 0
        num = 0
        for dataloader0 in tqdm(train_dataloader0):
                a, labela, p, labelp, n, labeln = dataloader0
                # train model
                # train model
                a = a.requires_grad_().to(opt.device)
                p = p.requires_grad_().to(opt.device)
                n = n.requires_grad_().to(opt.device)
                score, va, vp, vn, ca, cp, cn = model(a, p, n)
                target_ca,target_cp,target_cn = labela.to(opt.device),\
                                                labelp.to(opt.device),\
                                                labeln.to(opt.device)
                B, _, _, _ = a.shape
                target_bce = torch.cat((torch.ones(B), torch.zeros(B))).to(opt.device)
                optimizer.zero_grad()

                loss0 = criterion0(va,vp,vn)
                loss1 =  criterion1(ca, target_ca)\
                        +criterion1(cp, target_cp)\
                        +criterion1(cn, target_cn)
                loss2 =criterion2(score.squeeze(1),target_bce)
                loss = 0.3*loss0 + 0.05*loss1 + loss2
                loss.backward()
                optimizer.step()
                running_loss0 += loss0.item()
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                running_loss += loss.item()
                num += target_ca.shape[0]
        running_loss /= num
        running_loss0 /= num
        running_loss1 /= num
        running_loss2 /= num
        print("0.3*triplet_loss(margin = 0.8):",running_loss0,
              "| 0.05*ce_loss:",running_loss1,
              "|1*bce_loss:",running_loss2, running_loss)

        # update learning rate
        scheduler.step(running_loss)
        # validate
        MAP = 0
        print("FixedVector:   Youtube350")
        MAP += val_slow_multi(model, val_dataloader350, epoch)
        print("FixedVector:   Covers80")
        MAP += val_slow_multi(model, val_dataloader80, epoch)
        if MAP > best_MAP :
            best_MAP = MAP
            print('*****************BEST_MAP--FixedVector*****************')
            if parallel is True:
                model.module.save(opt.notes)
            else:
                model.save(opt.notes)
        MAP1 = 0
        # print("Sequence:      Youtube350")
        # MAP1 += val_slow_siamese(model, val_dataloader350,batch=80)
        print("Sequence:      Covers80")
        MAP1 += val_slow_siamese(model, val_dataloader80 ,batch=80)
        if MAP1 > best_MAP1:
            best_MAP1 = MAP1
            print('*****************BEST_MAP--Sequence*****************')
        # val_slow(model, val_dataloader2000, epoch)

        model.train()

def multiloss_train2(**kwargs):

    parallel = False
    margin = 0.9
    # 多尺度图片训练 396+
    # opt.feature, opt.model, opt.notes = 'cens', 'SPPNet','cens'
    opt._parse(kwargs)
    GPU = opt.GPU
    opt.notes = '400_train'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    opt.load_latest = False
    # opt.load_model_path = 'check_points/<class \'models.ATTSPPNet.ATTSPPNetV1\'>V1/1115_21:41:19.pth'
    data_length = opt.data_length
    test_data_length = opt.test_data_length
    # step1: configure model
    model = getattr(models, opt.model)()  # 从models文件夹里选出名字叫opt.model的模型
    if parallel is True:
        print("aa")
        model = torch.nn.DataParallel(model)
    if parallel is True:
        if opt.load_latest is True:
            model.module.load_latest(opt.notes)
        elif opt.load_model_path:
            model.module.load(opt.load_model_path)
    else:
        if opt.load_latest is True:
            model.load_latest(opt.notes)
        elif opt.load_model_path:
            model.load(opt.load_model_path)
    model.to(opt.device)
    # step1: configure model
    # step2: data

    train_data0 = get_triloader(out_length=400)
        # train_data1 = get_triloader(out_length=300)
        # train_data2 = get_triloader(out_length=400)
    val_data350 = CQT('songs350', out_length=400)
    val_data80 = CQT('songs80', out_length=400)
        # val_data = CQT('val', out_length=None)
        # test_data = CQT('test', out_length=None)
    train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # train_dataloader1 = DataLoader(train_data1, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # train_dataloader2 = DataLoader(train_data2, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=1)
    # test_dataloader = DataLoader(test_data, 1, shuffle=False, num_workers=1)
    val_dataloader80 = DataLoader(val_data80, 100, shuffle=False, num_workers=1)
    val_dataloader350 = DataLoader(val_data350, 100, shuffle=False, num_workers=1)
    # val_dataloader2000 = DataLoader(val_data2000, 1, shuffle=False, num_workers=1)
    # step3: criterion and optimizer
    criterion0 = torch.nn.TripletMarginLoss(margin=margin)
    criterion1 = torch.nn.CrossEntropyLoss()
    # criterion2 = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    if parallel is True:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=opt.lr_decay, patience=2, verbose=True, min_lr=5e-6)
    # train
    best_MAP = 0
    best_MAP1 = 0

    for epoch in range(opt.max_epoch):
        running_loss, running_loss0, running_loss1,running_loss2 = 0, 0, 0, 0
        num = 0
        for dataloader0 in tqdm(train_dataloader0):
                a, labela, p, labelp, n, labeln = dataloader0
                # train model
                # train model
                a = a.requires_grad_().to(opt.device)
                p = p.requires_grad_().to(opt.device)
                n = n.requires_grad_().to(opt.device)
                score, va, vp, vn, ca, cp, cn = model(a, p, n)
                target_ca,target_cp,target_cn = labela.to(opt.device),\
                                                labelp.to(opt.device),\
                                                labeln.to(opt.device)
                B, _, _, _ = a.shape
                target_bce = torch.cat((torch.ones(B), torch.zeros(B))).to(opt.device)
                optimizer.zero_grad()

                loss0 = criterion0(va,vp,vn)
                loss1 =  criterion1(ca, target_ca)\
                        +criterion1(cp, target_cp)\
                        +criterion1(cn, target_cn)
                loss2 =criterion1(score,target_bce.long())
                loss = 0.2*loss0 + 0.1*loss1 + loss2
                loss.backward()
                optimizer.step()
                running_loss0 += loss0.item()
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                running_loss += loss.item()
                num += target_ca.shape[0]
        running_loss /= num
        running_loss0 /= num
        running_loss1 /= num
        running_loss2 /= num
        print("0.3*triplet_loss(margin = 0.8):",running_loss0,
              "| 0.05*ce_loss:",running_loss1,
              "|1*bce_loss:",running_loss2, running_loss)

        # update learning rate
        scheduler.step(running_loss)
        # validate
        MAP = 0
        print("FixedVector:   Youtube350")
        MAP += val_slow_multi(model, val_dataloader350, epoch)
        print("FixedVector:   Covers80")
        MAP += val_slow_multi(model, val_dataloader80, epoch)
        if MAP > best_MAP :
            best_MAP = MAP
            print('*****************BEST_MAP--FixedVector*****************')
            if parallel is True:
                model.module.save(opt.notes)
            else:
                model.save(opt.notes)
        MAP1 = 0
        # print("Sequence:      Youtube350")
        # MAP1 += val_slow_siamese(model, val_dataloader350,batch=80)
        print("Sequence:      Covers80")
        MAP1 += val_slow_siamese(model, val_dataloader80 ,batch=200)
        if MAP1 > best_MAP1:
            best_MAP1 = MAP1
            print('*****************BEST_MAP--Sequence*****************')
        # val_slow(model, val_dataloader2000, epoch)

        model.train()
def test(**kwargs):
    opt.notes = '400_train'
    kwargs = {'model': 'SMCNN_4', 'batch_size': 16}
    opt.feature = 'cqt'
    opt.batch_size = 16
    opt._parse(kwargs)
    model = getattr(models, opt.model)()
    p = 'check_points/' + model.model_name + opt.notes
    if kwargs['model'] == 'SMCNN_4':
        f = os.path.join(p, "1227_08:47:43.pth")#'1218_15:02:04.pth'
    opt.load_model_path = "1227_08_47_43.pth"
    model.load(opt.load_model_path)
    model.to(opt.device)
    val_data80 = CQT('songs80', out_length=400)
    val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
    val_slow_multi(model, val_dataloader80, 0)
    val_slow_siamese2(model, val_dataloader80)
import copy

@torch.no_grad()
def val_slow(model, dataloader, epoch):
    model.eval()
    total, correct = 0, 0
    labels, features = None, None

    for ii, (data, label) in enumerate(dataloader):
        input = data.to(opt.device)
        # print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels, label))
        else:
            features = feature
            labels = label
    features = norm(features)

    z = copy.deepcopy(features.T)
    # dis2d = get_dis2d4(features)
    dis2d = -np.matmul(features, z)  # [-1,1] Because normalized, so mutmul is equal to ED

    np.save('dis80.npy', dis2d)
    np.save('label80.npy', labels)
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [100, 350])
    # elif len(labels) == 160:    MAP, top10, rank1 = calc_MAP(dis2d, labels,[80, 160])
    else:
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    # if MAP > best_MAP:
    #    best_MAP = MAP
    #    model.save('best_model')
    #    with open('check_points/'+opt.model+'/best.txt','w') as f:
    #        f.write('%f 10000 no weight'%MAP)
    #    print('best epoch!')
    print(epoch, MAP, top10, rank1)
    model.train()
    return MAP

@torch.no_grad()
def val_slow_multi(model, dataloader, epoch):
    model.eval()
    total, correct = 0, 0
    labels, features = None, None

    for ii, (data, label) in enumerate(dataloader):
        input = data.to(opt.device)
        # print(input.shape)
        feature = model.get_fixed_out(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels, label))
        else:
            features = feature
            labels = label
    features = norm(features)

    z = copy.deepcopy(features.T)
    # dis2d = get_dis2d4(features)
    dis2d = -np.matmul(features, z)  # [-1,1] Because normalized, so mutmul is equal to ED

    np.save('dis80.npy', dis2d)
    np.save('label80.npy', labels)
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [100, 350])
    # elif len(labels) == 160:    MAP, top10, rank1 = calc_MAP(dis2d, labels,[80, 160])
    else:
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    # if MAP > best_MAP:
    #    best_MAP = MAP
    #    model.save('best_model')
    #    with open('check_points/'+opt.model+'/best.txt','w') as f:
    #        f.write('%f 10000 no weight'%MAP)
    #    print('best epoch!')
    print(epoch, MAP, top10, rank1)
    model.train()
    return MAP
@torch.no_grad()
def val_slow_siamese(softdtw, dataloader, batch=20, is_dis='False'):
    softdtw.eval()
    if torch.cuda.device_count() > 1:
        softdtw.module.model.eval()
    else:
        softdtw.model.eval()
    seqs, labels = [], []
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        input = data.cuda()
        # _, seq, _ = softdtw.model(input)
        seqs.append(input)
        labels.append(label)
    seqs = torch.cat(seqs, dim=0)
    labels = torch.cat(labels, dim=0)

    N = labels.shape[0]
    if N == 350:
        query_l = [i // 100 for i in range(100 * 100, 350 * 100)]
        ref_l = [i for i in range(100)] * 250
    else:
        query_l = [i // N for i in range(N * N)]
        ref_l = [i for i in range(N)] * N
    dis2d = np.zeros((N, N))

    N = N * N if N != 350 else 100 * 250
    for st in range(0, N, batch):
        fi = (st + batch) if st + batch <= N else N
        query = seqs[query_l[st: fi], :, :]
        ref = seqs[ref_l[st: fi], :, :]
        if torch.cuda.device_count() > 1:
            s = softdtw.module.multi_compute_s(query, ref).data.cpu().numpy()
        else:
            s = softdtw.multi_compute_s(query, ref).data.cpu().numpy()
        for k in range(st, fi):
            i, j = query_l[k], ref_l[k]
            # print(i, j)
            if is_dis:
                dis2d[i, j] = s[k - st]
            else:
                dis2d[i, j] = -s[k - st]

    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [100, 350])
    else:
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    print(MAP, top10, rank1)

    softdtw.train()
    if torch.cuda.device_count() > 1:
        softdtw.module.model.train()
    else:
        softdtw.model.train()
    return MAP


@torch.no_grad()
def val_slow_multi2(model, dataloader, epoch):
    model.eval()
    total, correct = 0, 0
    labels, features = None, None

    for ii, (data, label) in enumerate(dataloader):
        input = data.to(opt.device)
        # print(input.shape)
        feature = model.get_fixed_out(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels, label))
        else:
            features = feature
            labels = label
    features = norm(features)

    z = copy.deepcopy(features.T)
    # dis2d = get_dis2d4(features)
    dis2d = -np.matmul(features, z)  # [-1,1] Because normalized, so mutmul is equal to ED

    np.save('dis80.npy', dis2d)
    np.save('label80.npy', labels)
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [100, 350])
    # elif len(labels) == 160:    MAP, top10, rank1 = calc_MAP(dis2d, labels,[80, 160])
    else:
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    # if MAP > best_MAP:
    #    best_MAP = MAP
    #    model.save('best_model')
    #    with open('check_points/'+opt.model+'/best.txt','w') as f:
    #        f.write('%f 10000 no weight'%MAP)
    #    print('best epoch!')
    print(epoch, MAP, top10, rank1)
    model.train()
    return MAP, top10, rank1


@torch.no_grad()
def val_slow_siamese2(softdtw, dataloader, style='null'):
    softdtw.eval()
    softdtw.model.eval()
    print('---get in---')
    seqs, labels = [], []
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        input = data

        seqs.append(input)
        labels.append(label)
    labels = torch.cat(labels, dim=0)

    N = labels.shape[0]
    if N == 350:
        query_l = [i // 100 for i in range(100 * 100, 350 * 100)]
        ref_l = [i for i in range(100)] * 250
    else:
        query_l = [i // N for i in range(N * N)]
        ref_l = [i for i in range(N)] * N
    dis2d = np.zeros((N, N))

    for st in range(0, N * N if N != 350 else 100 * 250):
        query = seqs[query_l[st]]
        ref = seqs[ref_l[st]]
        if style == 'min':
            T = min(query.shape[1], ref.shape[1])
            query, ref = query[:, :T, :], ref[:, :T, :]
        # print(softdtw.metric(query, ref))

        s = softdtw.multi_compute_s(query, ref)

        s = s.data.cpu().numpy()
        i, j = query_l[st], ref_l[st]
        dis2d[i, j], dis2d[j, i] = -s[0], -s[0]
    print('---calculate---')
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [100, 350])
    else:
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    print(MAP, top10, rank1)

    softdtw.train()
    softdtw.model.train()
    return MAP, top10, rank1
@torch.no_grad()
def val_quick(model, dataloader, note=None):
    print('-----------------------------')
    model.eval()
    total, correct = 0, 0
    features, labels = np.zeros([len(dataloader), 300]), np.zeros(len(dataloader))
    # features, labels = None,None
    time1 = time.time()
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        input = data.to(opt.device)

        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        features[ii], labels[ii] = feature, label  # only used when batch_size=1, otherwise use the code below
        '''
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels,label))
        else:
            features = feature
            labels = label
        '''
    features = norm(features)
    # time2=time.time()
    # print( 'network time: %.2fs' % (time2 - time1))
    # time2=time.time()
    dis = np.matmul(features, features.T)
    # time3=time.time()
    # print( 'claculate distance time: %.2fs' % (time3 - time2))
    # print(features.shape)
    # print(dis.shape)
    path_dir = 'hpcp/10'
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    features = features.astype(np.float32)
    features.tofile(path_dir + '/tmp_data.bin')
    np.savetxt(path_dir + '/tmp_ver.txt', labels, fmt='%d')
    thread_num = 30
    if len(labels) == 350:
        os.system('multi_map/main_test %s 100 350 300 %d' % (path_dir, thread_num))
    else:
        os.system('multi_map/main_test %s %d %d 300 %d' % (path_dir, len(labels), len(labels), thread_num))
    model.train()
if __name__=='__main__':
    import fire
    fire.Fire()