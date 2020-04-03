from data_loader import *
from torch.utils.data import DataLoader
from loss_mods import *
from tqdm import tqdm
from utility import *
import copy

from config import opt

torch.backends.cudnn.benchmark = True


def train(multi, loss_type, **kwargs):
    parallel = False
    margin = 0.9
    opt._parse(kwargs)
    GPU = opt.GPU
    opt.notes = 'Angle_train'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    opt.load_latest = False

    model = SMCNN_4_Angular(loss_type)
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

    train_data0 = get_triloader(out_length=400)
    val_data80 = CQT('songs80', out_length=400)
    train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader80 = DataLoader(val_data80, 100, shuffle=False, num_workers=1)
    #val_data350 = CQT('songs350', out_length=400)
    #val_dataloader350 = DataLoader(val_data350, 100, shuffle=False, num_workers=1)

    criterion0 = torch.nn.TripletMarginLoss(margin=margin)
    criterion1 = torch.nn.CrossEntropyLoss()

    lr = opt.lr
    if parallel is True:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=opt.lr_decay, patience=2, verbose=True, min_lr=5e-6)

    best_MAP = 0
    best_MAP1 = 0

    for epoch in range(opt.max_epoch):
        running_loss, running_loss0, running_loss1, running_loss2 = 0, 0, 0, 0
        num = 0
        for dataloader0 in tqdm(train_dataloader0):
            a, labela, p, labelp, n, labeln = dataloader0

            # train model
            a = a.requires_grad_().to(opt.device)
            p = p.requires_grad_().to(opt.device)
            n = n.requires_grad_().to(opt.device)
            target_ca, target_cp, target_cn = labela.to(opt.device), \
                                              labelp.to(opt.device), \
                                              labeln.to(opt.device)
            score, va, vp, vn, ang_loss = model(a, p, n, target_ca, target_cp, target_cn)
            B, _, _, _ = a.shape
            target_bce = torch.cat((torch.ones(B), torch.zeros(B))).to(opt.device)
            optimizer.zero_grad()

            loss0 = criterion0(va, vp, vn)

            loss1 = ang_loss
            loss2 = criterion1(score, target_bce.long())
            if multi:
                loss = 0.2 * loss0 + 0.1 * loss1 + loss2
            else:
                loss = loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss0 += loss0.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss += loss.item()
            num += target_ca.shape[0]

        print(running_loss)

        running_loss /= num

        scheduler.step(running_loss)
        '''
        # validate
        MAP = 0
        print("FixedVector:   Youtube350")
        MAP += val_slow_multi(model, val_dataloader350, epoch)
        print("FixedVector:   Covers80")
        MAP += val_slow_multi(model, val_dataloader80, epoch)
        if MAP > best_MAP:
            best_MAP = MAP
            print('*****************BEST_MAP--FixedVector*****************')
            if parallel is True:
                model.module.save(opt.notes)
            else:
                model.save(opt.notes)
        MAP1 = 0
        print("Sequence:      Youtube350")
        MAP1 += val_slow_siamese(model, val_dataloader350,batch=80)
        print("Sequence:      Covers80")
        MAP1 += val_slow_siamese(model, val_dataloader80, batch=200)        if MAP1 > best_MAP1:
            best_MAP1 = MAP1
            print('*****************BEST_MAP--Sequence*****************')
        val_slow(model, val_dataloader2000, epoch)
        '''
        model.train()


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


def check_lt(loss_type):
    if loss_type != 'lsoftmax' and loss_type != 'sphereface' and loss_type != 'arcface' and loss_type != 'cosface':
        return False
    else:
        return True


if __name__ == '__main__':
    argc = len(sys.argv)
    multi = False
    loss_type = 'lsoftmax'
    for i in range(1, argc):
        if sys.argv[i][0] == '-':
            if sys.argv[i][1] == 'm':
                multi = True
            elif sys.argv[i][1] == 'l':
                i = i + 1
                if check_lt(sys.argv[i]):
                    loss_type = sys.argv[i]
                else:
                    print('Wrong loss type!!')
                    exit()

    train(multi, loss_type)