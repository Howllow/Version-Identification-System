from data_loader import *
from torch.utils.data import DataLoader
from loss_mods import *
from tqdm import tqdm
from utility import *
import copy
import warnings
from torchvision import datasets, transforms
import models
from models.resnet import *

import sys

from config import opt

torch.backends.cudnn.benchmark = True

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def train(multi, loss_type, **kwargs):
    parallel = False
    loss_path = "./data/loss_out_" + loss_type + ".txt"
    margin = 0.9
    opt._parse(kwargs)
    GPU = opt.GPU
    opt.notes = 'Angle_train_' + loss_type
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    opt.load_latest = False

    model = SMCNN_4_Angular()
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
    train_data1 = CQT('train', out_length=400)
    val_data = CQT('songs80', out_length=400)
    train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    val_dataloader = DataLoader(val_data, 100, shuffle=False, num_workers=1)
    #val_data350 = CQT('songs350', out_length=400)
    #val_dataloader350 = DataLoader(val_data350, 100, shuffle=False, num_workers=1)

    criterion0 = torch.nn.TripletMarginLoss(margin=margin).to(opt.device)
    criterion1 = torch.nn.CrossEntropyLoss().to(opt.device)
    criterion2 = CircleLoss(m=0.25, gamma=64).to(opt.device)
    criterion3 = SphereLoss().to(opt.device)
    criterion4 = torch.nn.BCELoss().to(opt.device)

    lsoftmax = LSoftmaxLinear(300, 10000, 2, 'cuda').to(opt.device)
    sphereface = SphereFace(300, 10000).to(opt.device)
    arcface = ArcMarginProduct(300, 10000).to(opt.device)

    if loss_type == 'cosface':
        criterion3 = AngularPenaltySMLoss(300, 10000, loss_type).to(opt.device)

    lr = opt.lr
    if parallel is True:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    if loss_type == 'lsoftmax':
        optimizer = torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': lsoftmax.parameters()}
                ], lr=opt.lr, weight_decay=opt.weight_decay)

    elif loss_type == 'sphereface':
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': sphereface.parameters()}
        ], lr=opt.lr, weight_decay=opt.weight_decay)

    elif loss_type == 'cosface':
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': criterion3.parameters()}
        ], lr=opt.lr, weight_decay=opt.weight_decay)

    elif loss_type == 'arcface':
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': arcface.parameters()}
        ], lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=opt.lr_decay, patience=4, verbose=True, min_lr=5e-6)

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
            score, va, vp, vn, la, lp, ln, da, dp, dn = model(a, p, n, target_ca, target_cp, target_cn)
            B, _, _, _ = a.shape
            target_bce = torch.cat((torch.ones(B), torch.zeros(B))).to(opt.device)

            loss0 = criterion0(va, vp, vn)
            loss2 = criterion1(score, target_bce.long())

            if loss_type == 'lsoftmax':
                loss1 = criterion1(lsoftmax(va, target_ca), target_ca)\
                        + criterion1(lsoftmax(vp, target_cp), target_cp)\
                        + criterion1(lsoftmax(vn, target_cn), target_cn)

            elif loss_type == 'sphereface':
                loss1 = criterion3(sphereface(va, target_ca), target_ca)\
                        + criterion3(sphereface(vp, target_cp), target_cp)\
                        + criterion3(sphereface(vn, target_cn), target_cn)

            elif loss_type == 'arcface':
                loss1 = criterion1(arcface(va, target_ca), target_ca)\
                        + criterion1(arcface(vp, target_cp), target_cp)\
                        + criterion1(arcface(vn, target_cn), target_cn)

            elif loss_type == 'base':
                loss1 = criterion1(la, target_ca)\
                        + criterion1(lp, target_cp)\
                        + criterion1(ln, target_cn)

            elif loss_type != 'circlelabel':
                loss1 = criterion3(va, target_ca) + criterion3(vp, target_cp) + criterion3(vn, target_cn)

            if loss_type == 'circlelabel':
                feats = torch.cat((va, vp, vn), 0)
                labels = torch.cat((target_ca, target_cp, target_cn), 0)
                loss = criterion2(*convert_label_to_similarity(nn.functional.normalize(feats), labels))

            elif multi:
                loss = 0.2 * loss0 + 0.1 * loss1 + loss2
                running_loss1 += loss1.item()
                running_loss0 += loss0.item()
                running_loss2 += loss2.item()

            elif loss_type == 'base':
                loss = loss2

            else:
                loss = loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num += target_ca.shape[0]

        if multi:
            print("loss:")
            print(running_loss0, running_loss1, running_loss2)
        else:
            print("loss:")
            print(running_loss)

        running_loss /= num

        scheduler.step(running_loss)


        # validate
        if epoch:
            MAP = 0
            MAP += val_slow_multi(model, val_dataloader, epoch)
            if MAP > best_MAP:
                best_MAP = MAP
                print('*****************BEST_MAP--FixedVector*****************')
                if parallel is True:
                    model.module.save(opt.notes)
                else:
                    model.save(opt.notes)


@torch.no_grad()
def val_slow_multi(model, dataloader, epoch):
    model.eval()
    total, correct = 0, 0
    labels, features = None, None

    for ii, (data, label, _) in enumerate(dataloader):
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
def val_slow_siamese(softdtw, dataloader, batch=20, is_dis=False):
    softdtw.eval()
    if torch.cuda.device_count() > 1:
        softdtw.module.model.eval()
    else:
        softdtw.model.eval()
    seqs, labels = [], []
    for ii, (data, label, _) in tqdm(enumerate(dataloader)):
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
    if loss_type != 'lsoftmax' and loss_type != 'sphereface' and loss_type != 'arcface' and loss_type != 'cosface' and \
            loss_type != 'circlepair' and loss_type != 'circlelabel' and loss_type != 'base':
        return False
    else:
        return True


def mnist_try(loss_type, **kwargs):
    warnings.filterwarnings('ignore')
    opt._parse(kwargs)
    GPU = opt.GPU
    opt.notes = '400_train'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    opt.load_latest = False

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    lsoftmax = LSoftmaxLinear(3, 10, 4, 'cuda').to(opt.device)
    sphereface = SphereFace(3, 10).to(opt.device)
    arcface = ArcMarginProduct(2, 10).to(opt.device)
    cosface = AddMarginProduct(2, 10).to(opt.device)


    model = MNISTNet().to(opt.device)

    if loss_type == 'lsoftmax':
        criterion = nn.CrossEntropyLoss().to(opt.device)
        optimizer = torch.optim.SGD([
                    {'params': model.parameters()},
                    {'params': lsoftmax.parameters()}
                ], lr=opt.lr, momentum=0.9)

    elif loss_type == 'sphereface':
        criterion = SphereLoss().to(opt.device)
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': sphereface.parameters()}
        ], lr=opt.lr)

    elif loss_type == 'arcface':
        criterion = AngularPenaltySMLoss(3, 10, 'arcface').to(opt.device)
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': criterion.parameters()}
        ], lr=0.01)

    elif loss_type == 'cosface':
        criterion = AngularPenaltySMLoss(3, 10, 'cosface').to(opt.device)
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': criterion.parameters()}
        ], lr=0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=opt.lr_decay, patience=2, verbose=True, min_lr=5e-6)

    for epoch in range(1, opt.max_epoch):
        model.train()
        running_loss, num = 0, 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(opt.device), target.to(opt.device)
            output = model(data)
            if loss_type == 'lsoftmax':
                logit = lsoftmax(output, target)
                loss = criterion(logit, target)

            elif loss_type == 'sphereface':
                loss = criterion(sphereface(output, target), target)

            elif loss_type == 'arcface':
                loss = criterion(output, target)

            elif loss_type == 'cosface':
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num += target.shape[0]
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        running_loss /= num
        scheduler.step(running_loss)

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(opt.device), target.to(opt.device)
                output = model(data)
                if loss_type == 'lsoftmax':
                    logit = lsoftmax(output, target)
                    test_loss += criterion(logit, target).item()
                    pred = logit.max(1, keepdim=True)[1]

                elif loss_type == 'sphereface':
                    res = sphereface(output, target)
                    test_loss += criterion(res, target).item()
                    pred = torch.max(res[0].data, 1)[1]

                elif loss_type == 'arcface':
                    test_loss += criterion(output, target).item()

                elif loss_type == 'cosface':
                    test_loss += criterion(output, target).item()

                #correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def model_test(**kwargs):
    opt._parse(kwargs)
    model = SMCNN_4_Angular('cosface').to(opt.device)
    model.load("./templates/0428_03:26:55.pth")
    model1 = getattr(models, 'SMCNN_4')().to(opt.device)
    model1.load("1227_08_47_43.pth")
    val_data = CQT('songs80', out_length=400)
    val_dataloader = DataLoader(val_data, 100, shuffle=False, num_workers=1)
    val_slow_multi(model, val_dataloader, 1)
    val_slow_siamese(model, val_dataloader, 20)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    argc = len(sys.argv)
    multi = False
    loss_type = 'lsoftmax'
    mnist = False
    good = False

    for i in range(1, argc):
        if sys.argv[i][0] == '-':
            if sys.argv[i][1] == 't':
                mnist = True
            elif sys.argv[i][1] == 'm':
                multi = True
            elif sys.argv[i][1] == 'g':
                good = True
            elif sys.argv[i][1] == 'l':
                i = i + 1
                if check_lt(sys.argv[i]):
                    loss_type = sys.argv[i]
                else:
                    print('Wrong loss type!!')
                    exit()

    if good:
        model_test()

    elif mnist:
        mnist_try(loss_type)

    else:
        train(multi, loss_type)
