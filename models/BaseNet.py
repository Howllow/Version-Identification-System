import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import math
from .BasicModule import BasicModule
from .FCN import *

class BaseNet(BasicModule):
    def __init__(self, num_init_features=512, num_classes=10000):
        super(BaseNet, self).__init__()
        # input [N, C, H, W]
        # First convolution
        # 频带卷积结构，输入nx23,经过75x12的卷积核，得到[C=64, H=12, W=n-d+1]的特征
        # 再经过12x1的池化操作得到[64, 1, n-d+1 x1 ]
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=num_init_features,
                                kernel_size=(12, 75), stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=(12, 1), stride=(1, 1))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=num_init_features, out_channels=1024,
                                kernel_size=(1, 3), stride=1, bias=False)),
            ('norm1', nn.BatchNorm2d(1024)),
            ('relu1', nn.ReLU(inplace=True)),
            # ('pool0', nn.AvgPool2d(kernel_size=(12,1), stride=(1,1))),
        ]))
        # self.spp = SPP([32,16,10,8,6,4,2,1])
        self.fc0 = nn.Linear(1024, 2048)
        self.fc1 = nn.Linear(2048, num_classes)

    def forward(self, x):
        # input [N, C, H, W] (W = 396)
        x = self.features(x)  # [N, 512, 1, W - 75 + 1]
        x = self.conv(x)  # [N, 1024, 1, W - 75 +1 - 3 + 1]
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size()[0], -1)  # [N, 1024]
        feature = F.relu(self.fc0(x))
        x = self.fc1(feature)

        return x, feature


class CQTBaseNet(BasicModule):
    def __init__(self):
        super().__init__()
        # input N, C, 72, L
        # First convolution
        # 频带卷积结构，输入nx84,经过75x12的卷积核，得到[N, C, 61, L]的特征
        # 再经过12x1的池化操作得到[64, 1, n-d+1 x1 ]
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(36, 75),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12, 3),
                                stride=(1, 1), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                stride=(1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), bias=False)),
            ('norm0', nn.BatchNorm2d(256)),
            ('relu0', nn.ReLU(inplace=True)),

            ('conv1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 3), bias=False)),
            ('norm1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU(inplace=True)),

            ('conv2', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1, 3), bias=False)),
            ('norm2', nn.BatchNorm2d(1024)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, 32))),
        ]))
        # self.spp = SPP([32,16,10,8,6,4,2,1])
        self.fc0 = nn.Linear(1024 * 32, 300)
        self.fc1 = nn.Linear(300, 10000)

    def forward(self, x):
        # input [N, C, H, W] (W = 396)
        N = x.size()[0]
        x = self.features(x)  # [N, 128, 1, W - 75 + 1]
        x32 = self.conv(x)  # [N, 256, 1, W - 75 +1 - 3 + 1]
        # x = SPP(x, [32,16,10,8,6,4,2,1]) # [N, 256, 1, sum()=79]
        x = x32.view(N, -1)
        feature = self.fc0(x)
        x = self.fc1(feature)
        return x, feature, x32

class Temporal_Inception(BasicModule):
    def __init__(self):
        super().__init__()
        self.features1 = nn.Sequential(OrderedDict([
            ('tnc1', temporal_inception(in_channels=1, out_channels=32,
                                        kernel_size=(12, 3), inter_channels=32)),
            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
            ('tnc2', temporal_inception(in_channels=32, out_channels=64,
                                        kernel_size=(13, 3), inter_channels=32)),
            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
            ('tnc3', temporal_inception(in_channels=64, out_channels=128,
                                        kernel_size=(13, 3), inter_channels=64)),
            ('maxpool3', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
            ('tnc4', temporal_inception(in_channels=128, out_channels=256,
                                        kernel_size=(3, 3), inter_channels=128)),
            ('maxpool4', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
            ('tnc5', temporal_inception(in_channels=256, out_channels=512,
                                        kernel_size=(3, 3), inter_channels=256)),
            ('maxpool5', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
            ('tnc6', temporal_inception(in_channels=512, out_channels=512,
                                        kernel_size=(3, 3), inter_channels=256)),
            ('maxpool6', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
        ]))

        self.features1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(12, 3),
                                dilation=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),

            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True))
        ]))
        # ('maxpool0',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

        self.features2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),

            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        self.features3 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),

            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        self.features4 = nn.Sequential(OrderedDict([
            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        self.features5 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None)))
        ]))

    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)

        x1 = nn.AdaptiveMaxPool2d((1, None))(x1).squeeze(dim=2).permute(0, 2, 1)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2).permute(0, 2, 1)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2).permute(0, 2, 1)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2).permute(0, 2, 1)
        x5 = x5.squeeze(dim=2).permute(0, 2, 1)
        return x2, x3, x4, x5
class temporal_inception(BasicModule):
    def __init__(self,in_channels,inter_channels,out_channels,kernel_size):
        super().__init__()
        self.conv1_3 = nn.Conv2d(in_channels=in_channels,out_channels=inter_channels,
                                 kernel_size=kernel_size,padding=(1,1), dilation=(1,1))
        self.conv1_5 = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                                 kernel_size=kernel_size,padding=(1,2), dilation=(1,2))
        self.conv1_7 = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                                 kernel_size=kernel_size,padding=(1,3), dilation=(1,3))
        self.conv1_1 = nn.Conv2d(in_channels=inter_channels*3, out_channels=out_channels,kernel_size=(1,1))
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(out_channels)
    def get_feature(self,x):
        x1 = self.conv1_3(x)
        x2 = self.conv1_5(x)
        x3 = self.conv1_7(x)
        x = torch.cat((x1, x2, x3), 1)
        x = self.conv1_1(x)
        x = self.bn(x)
        return x1,x2,x3,x
    def forward(self, x):
        x1 = self.conv1_3(x)
        x2 = self.conv1_5(x)
        x3 = self.conv1_7(x)
        x = torch.cat((x1,x2,x3),1)
        x = self.conv1_1(x)
        x = self.bn(x)
        return self.relu(x)
class CQT_inception(BasicModule):
    def __init__(self ):
        super().__init__()
        self.features1 = nn.Sequential(OrderedDict([
            ('tnc1',temporal_inception(in_channels=1,out_channels=32,
                                       kernel_size=(12,3),inter_channels=32)),
            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
            ('tnc2', temporal_inception( in_channels=32,out_channels=64,
                                        kernel_size=(13, 3), inter_channels=32)),
            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
            ('tnc3', temporal_inception(in_channels=64, out_channels=128,
                                        kernel_size=(13, 3), inter_channels=64)),
            ('maxpool3', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
            ('tnc4', temporal_inception(in_channels=128, out_channels=256,
                                        kernel_size=(3, 3), inter_channels=128)),
            ('maxpool4', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
            ('tnc5', temporal_inception(in_channels=256, out_channels=512,
                                        kernel_size=(3, 3), inter_channels=256)),
            ('maxpool5', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
            ('tnc6', temporal_inception(in_channels=512, out_channels=512,
                                        kernel_size=(3, 3), inter_channels=256)),
            ('maxpool6', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
        ]))
        # ('maxpool0',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),
        self.fc1 = nn.Linear(512*3*6,512*4)
        self.fc2 = nn.Linear(512*4,512)
        self.fc3 = nn.Linear(512,10000)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(512)

    def forward(self, x):

        x = self.features1(x).view(x.size()[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        c = self.relu(self.fc3(x))
        return c,x
class CQT_inception_2(BasicModule):
    def __init__(self ):
        super().__init__()
        self.features1 = nn.Sequential(OrderedDict([
            ('tnc1',temporal_inception(in_channels=1,out_channels=32,
                                       kernel_size=(12,3),inter_channels=32)),
            ('tnc1', temporal_inception(in_channels=32, out_channels=32,
                                        kernel_size=(13, 3), inter_channels=32)),
            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),

            ('tnc2', temporal_inception( in_channels=32,out_channels=64,
                                        kernel_size=(13, 3), inter_channels=32)),
            ('tnc2', temporal_inception(in_channels=64, out_channels=64,
                                        kernel_size=(13, 3), inter_channels=64)),
            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),

            ('tnc3', temporal_inception(in_channels=64, out_channels=128,
                                        kernel_size=(3, 3), inter_channels=64)),
            ('tnc2', temporal_inception(in_channels=128, out_channels=128,
                                        kernel_size=(3, 3), inter_channels=128)),
            ('maxpool3', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),

            ('tnc4', temporal_inception(in_channels=128, out_channels=256,
                                        kernel_size=(3, 3), inter_channels=128)),
            ('tnc2', temporal_inception(in_channels=256, out_channels=256,
                                        kernel_size=(3, 3), inter_channels=256)),
            ('maxpool4', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),

            ('tnc5', temporal_inception(in_channels=256, out_channels=512,
                                        kernel_size=(3, 3), inter_channels=256)),
            ('tnc2', temporal_inception(in_channels=512, out_channels=512,
                                        kernel_size=(3, 3), inter_channels=512)),
            ('maxpool5', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
        ]))
        # ('maxpool0',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),
        self.fc1 = nn.Linear(512*3*6,512*4)
        self.fc2 = nn.Linear(512*4,512)
        self.fc3 = nn.Linear(512,10000)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.features1(x)
        print(x.shape)
        # x = self.features1(x).view(x.size()[0],-1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # c = self.relu(self.fc3(x))
        return x


class CQTSPPNet_seq_dilation_SPP(BasicModule):
    def __init__(self):
        super().__init__()
        self.features1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(12, 3),
                                dilation=(1, 1), bias=False,padding=(5,1))),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),

            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 2), bias=False,padding=(6,1))),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 1), bias=False,padding=(6,1))),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool0', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
        ]))
        # ('maxpool0',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

        self.features2 = nn.Sequential(OrderedDict([

            ('conv3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False,padding=(1,1))),
            ('norm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),

            ('maxpool1', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
        ]))
        self.features3 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False,padding=(1,1))),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False,padding=(1,1))),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),

            ('maxpool2', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
        ]))
        self.features4 = nn.Sequential(OrderedDict([
            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False,padding=(1,1))),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False,padding=(1,1))),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
        ]))
        self.features5 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False,padding=(1,1))),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False,padding=(1,1))),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('maxpool4', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
        ]))
        self.features6 = nn.Sequential(OrderedDict([
            ('conv10', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False, padding=(1, 1))),
            ('norm10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU(inplace=True)),
            ('conv11', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False, padding=(1, 1))),
            ('norm11', nn.BatchNorm2d(512)),
            ('relu11', nn.ReLU(inplace=True)),
            ('maxpool5', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
        ]))
    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]

        x2 = self.features2(x1)

        x3 = self.features3(x2)

        x4 = self.features4(x3)

        x5 = self.features5(x4)

        x6 = self.features6(x5)

        x1 = nn.AdaptiveMaxPool2d((1, None))(x1).squeeze(dim=2).permute(0, 2, 1)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2).permute(0, 2, 1)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2).permute(0, 2, 1)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2).permute(0, 2, 1)
        x5 = nn.AdaptiveMaxPool2d((1, None))(x5).squeeze(dim=2).permute(0, 2, 1)
        x6 = x6.view(x6.shape[0],-1)
        return x1, x2, x3, x4, x5,x6
class SMCNN(BasicModule):
    """
    55000it [75:34:19,  4.73s/it]train_loss: 0.007741250745826789
Youtube350:
                         0.9561629266311203 0.1928 1.876
CoverSong80:, 513.07it/s]
                         0.902245744393522 0.0925 4.0875
SH100K:
                         0.7396470873452758 0.49920356499478524 54.79321133971745
*****************BEST*****************
model name 0819_01:03:39.pth
    """
    def __init__(self ):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation_SPP()

        self.VGG_Conv1 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')

        self.fc = nn.Linear(42496, 1)
        self.fc2 = nn.Linear(2048,512)
        self.fc3 = nn.Linear(512,10000)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqp.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return d_ap_s if debug == False else d_ap_s
    def get_fixed_out(self,seqa):
        _, _, _, _,_,x6 = self.model(seqa)
        x6 = self.fc2(x6)
        return x6

    def multi_compute_s(self, seqa, seqb):
        seqa1, seqa2, seqa3, seqa4,_,_  = self.model(seqa)
        seqb1, seqb2, seqb3, seqb4,_,_ = self.model(seqb)
        # p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqb4).unsqueeze(1)
        # p_a = torch.cat((p_a2,p_a3,p_a4),3)
        # torch.Size([1, 1, 84, 400])
        # torch.Size([1, 194, 64])
        # torch.Size([1, 94, 128])
        # torch.Size([1, 44, 256])
        # torch.Size([1, 38, 512])
        VGG_out0 = self.VGG_Conv1(p_a1)
        VGG_out0 = VGG_out0['x5'].view(VGG_out0['x4'].shape[0], -1)
        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat((VGG_out0,VGG_out1,VGG_out2, VGG_out3), 1)
        samil = torch.sigmoid(self.fc(VGG_out))

        return samil

    def forward(self, seqa, seqb, seqn):
        seqa1, seqa2, seqa3, seqa4,seqa5,xa6 = self.model(seqa)

        seqb1, seqb2, seqb3, seqb4,seqb5,xb6 = self.model(seqb)
        p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqb4).unsqueeze(1)
        # p_a = torch.cat((p_a2,p_a3,p_a4),3)
        # torch.Size([1, 1, 84, 400])
        # torch.Size([1, 194, 64])
        # torch.Size([1, 94, 128])
        # torch.Size([1, 44, 256])
        # torch.Size([1, 38, 512])
        VGG_out0 = self.VGG_Conv1(p_a1)
        VGG_out0 = VGG_out0['x5'].view(VGG_out0['x4'].shape[0], -1)
        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat((VGG_out0, VGG_out1, VGG_out2, VGG_out3), 1)
        p_ap= torch.sigmoid(self.fc(VGG_out))


        seqn1, seqn2, seqn3, seqn4,seqn5,xn6 = self.model(seqn)
        n_a1 = self.metric(seqa1, seqn1).unsqueeze(1)
        n_a2 = self.metric(seqa2, seqn2).unsqueeze(1)
        n_a3 = self.metric(seqa3, seqn3).unsqueeze(1)
        n_a4 = self.metric(seqa4, seqn4).unsqueeze(1)
        VGG_out0 = self.VGG_Conv1(n_a1)
        VGG_out0 = VGG_out0['x5'].view(VGG_out0['x4'].shape[0], -1)
        VGG_out1 = self.VGG_Conv1(n_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(n_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(n_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat((VGG_out0, VGG_out1, VGG_out2, VGG_out3), 1)
        p_an = torch.sigmoid(self.fc(VGG_out))

        xa6,xb6,xn6= self.fc2(xa6),self.fc2(xb6),self.fc2(xn6)
        pca,pcb,pcn = self.fc3(xa6),self.fc3(xb6),self.fc3(xn6)
        return torch.cat((p_ap, p_an), dim=0),xa6,xb6,xn6,pca,pcb,pcn
class SMCNN_2(BasicModule):
    """
    55000it [75:34:19,  4.73s/it]train_loss: 0.007741250745826789
Youtube350:
                         0.9561629266311203 0.1928 1.876
CoverSong80:, 513.07it/s]
                         0.902245744393522 0.0925 4.0875
SH100K:
                         0.7396470873452758 0.49920356499478524 54.79321133971745
*****************BEST*****************
model name 0819_01:03:39.pth
    """
    def __init__(self ):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation_SPP()

        self.VGG_Conv1 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')

        self.fc = nn.Linear(24064, 1)
        self.fc2 = nn.Linear(2048,512)
        self.fc3 = nn.Linear(512,10000)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqp.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return d_ap_s if debug == False else d_ap_s
    def get_fixed_out(self,seqa):
        _, _, _, _,_,x6 = self.model(seqa)
        x6 = self.fc2(x6)
        return x6

    def multi_compute_s(self, seqa, seqb):
        _, seqa2, seqa3, seqa4,_,_  = self.model(seqa)
        _, seqb2, seqb3, seqb4,_,_ = self.model(seqb)
        # p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        # p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqb4).unsqueeze(1)
        # p_a = torch.cat((p_a2,p_a3,p_a4),3)
        # torch.Size([1, 1, 84, 400])
        # torch.Size([1, 194, 64])
        # torch.Size([1, 94, 128])
        # torch.Size([1, 44, 256])
        # torch.Size([1, 38, 512])
        # VGG_out0 = self.VGG_Conv1(p_a1)
        # VGG_out0 = VGG_out0['x5'].view(VGG_out0['x4'].shape[0], -1)
        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat((VGG_out1,VGG_out2, VGG_out3), 1)
        samil = torch.sigmoid(self.fc(VGG_out))

        return samil

    def forward(self, seqa, seqb, seqn):
        _, seqa2, seqa3, seqa4,_,xa6 = self.model(seqa)

        _, seqb2, seqb3, seqb4,_,xb6 = self.model(seqb)
        # p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqb4).unsqueeze(1)

        # VGG_out0 = self.VGG_Conv1(p_a1)
        # VGG_out0 = VGG_out0['x5'].view(VGG_out0['x4'].shape[0], -1)
        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat(( VGG_out1, VGG_out2, VGG_out3), 1)
        p_ap= torch.sigmoid(self.fc(VGG_out))


        _, seqn2, seqn3, seqn4,_,xn6 = self.model(seqn)
        # n_a1 = self.metric(seqa1, seqn1).unsqueeze(1)
        n_a2 = self.metric(seqa2, seqn2).unsqueeze(1)
        n_a3 = self.metric(seqa3, seqn3).unsqueeze(1)
        n_a4 = self.metric(seqa4, seqn4).unsqueeze(1)
        # VGG_out0 = self.VGG_Conv1(n_a1)
        # VGG_out0 = VGG_out0['x5'].view(VGG_out0['x4'].shape[0], -1)
        VGG_out1 = self.VGG_Conv1(n_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(n_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(n_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat(( VGG_out1, VGG_out2, VGG_out3), 1)
        p_an = torch.sigmoid(self.fc(VGG_out))

        xa6,xb6,xn6= self.fc2(xa6),self.fc2(xb6),self.fc2(xn6)
        pca,pcb,pcn = self.fc3(xa6),self.fc3(xb6),self.fc3(xn6)
        return torch.cat((p_ap, p_an), dim=0),xa6,xb6,xn6, pca,pcb,pcn


class CQTSPPNet_seq_dilation_SPP_2(BasicModule):
    def __init__(self):
        super().__init__()
        self.features1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(12, 3),
                                dilation=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),

            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True))
        ]))
        # ('maxpool0',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

        self.features2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),

            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        self.features3 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),

            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        self.features4 = nn.Sequential(OrderedDict([
            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        self.features5 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None)))
        ]))

    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)

        x1 = nn.AdaptiveMaxPool2d((1, None))(x1).squeeze(dim=2).permute(0, 2, 1)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2).permute(0, 2, 1)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2).permute(0, 2, 1)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2).permute(0, 2, 1)
        x5 = x5.squeeze(dim=2).permute(0, 2, 1)
        return x2, x3, x4, x5

class SMCNN_3(BasicModule):

    def __init__(self):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation_SPP_2()
        self.VGG_Conv1 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')
        self.fc = nn.Linear(18944, 1)
        self.adp_max_pool = torch.nn.AdaptiveMaxPool2d((1,None))
        self.fc1 = nn.Linear(512 + 256 + 128, 300)
        self.fc2 = nn.Linear(300, 10000)
        self.dropout = nn.Dropout(0.7)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqp.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)


        return d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self, seqa, seqb):
        seqa1, seqa2, seqa3, seqa4, = self.model(seqa)
        seqb1, seqb2, seqb3, seqb4 = self.model(seqb)
        # p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        # p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqb4).unsqueeze(1)
        # p_a = torch.cat((p_a2,p_a3,p_a4),3)
        # torch.Size([1, 1, 84, 400])
        # torch.Size([1, 194, 64])
        # torch.Size([1, 94, 128])
        # torch.Size([1, 44, 256])
        # torch.Size([1, 38, 512])
        # VGG_out0 = self.VGG_Conv1(p_a1)
        # VGG_out0 = VGG_out0['x5'].view(VGG_out0['x5'].shape[0], -1)

        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat((VGG_out1,VGG_out2, VGG_out3), 1)
        samil = torch.sigmoid(self.fc(VGG_out))

        return samil

    def get_fixed_out(self, seqa):
        seqa1, seqa2, seqa3, seqa4 = self.model(seqa)
        seqa2, seqa3, seqa4 = self.adp_max_pool(seqa2).squeeze(1), \
                              self.adp_max_pool(seqa3).squeeze(1), \
                              self.adp_max_pool(seqa4).squeeze(1)
        seqa = torch.cat((seqa2, seqa3, seqa4), 1)
        seqa = self.fc1(seqa)
        return seqa
    def forward(self, seqa, seqp, seqn):
        seqa1, seqa2, seqa3, seqa4 = self.model(seqa)
        seqp1, seqp2, seqp3, seqp4 = self.model(seqp)
        seqn1, seqn2, seqn3, seqn4 = self.model(seqn)

        p_a2 = self.metric(seqa2, seqp2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqp3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqp4).unsqueeze(1)

        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat((VGG_out1, VGG_out2, VGG_out3), 1)
        p_ap = torch.sigmoid(self.fc(VGG_out))

        p_b2 = self.metric(seqa2, seqn2).unsqueeze(1)
        p_b3 = self.metric(seqa3, seqn3).unsqueeze(1)
        p_b4 = self.metric(seqa4, seqn4).unsqueeze(1)
        VGG_out1 = self.VGG_Conv1(p_b2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_b3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_b4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat((VGG_out1, VGG_out2, VGG_out3), 1)
        p_an = torch.sigmoid(self.fc(VGG_out))
        seqa2, seqa3, seqa4 = self.adp_max_pool(seqa2).squeeze(1), \
                              self.adp_max_pool(seqa3).squeeze(1), \
                              self.adp_max_pool(seqa4).squeeze(1)
        seqa = torch.cat((seqa2, seqa3, seqa4), 1)
        seqp2, seqp3, seqp4 = self.adp_max_pool(seqp2).squeeze(1), \
                              self.adp_max_pool(seqp3).squeeze(1), \
                              self.adp_max_pool(seqp4).squeeze(1)
        seqp = torch.cat((seqp2, seqp3, seqp4), 1)
        seqn2, seqn3, seqn4 = self.adp_max_pool(seqn2).squeeze(1), \
                              self.adp_max_pool(seqn3).squeeze(1), \
                              self.adp_max_pool(seqn4).squeeze(1)
        seqn = torch.cat((seqn2, seqn3, seqn4), 1)
        seqa ,seqp, seqn = self.fc1(seqa), self.fc1(seqp), self.fc1(seqn)
        seqa_d,seqp_d,seqn_d=self.dropout(seqa),self.dropout(seqp),self.dropout(seqn)
        la, lp, ln = self.fc2(seqa_d), self.fc2(seqp_d), self.fc2(seqn_d)
        return torch.cat((p_ap, p_an), dim=0),seqa,seqp,seqn,la,lp,ln
class SMCNN_4(BasicModule):

    def __init__(self):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation_SPP_2()
        self.VGG_Conv1 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')
        self.fc = nn.Linear(18944, 2)
        self.adp_max_pool = torch.nn.AdaptiveMaxPool2d((1,None))
        self.fc1 = nn.Linear(512 + 256 + 128, 300)
        self.fc2 = nn.Linear(300, 10000)
        self.dropout = nn.Dropout(0.7)
        self.relu= torch.nn.ReLU()
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqp.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)


        return d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self, seqa, seqb):
        seqa1, seqa2, seqa3, seqa4, = self.model(seqa)
        seqb1, seqb2, seqb3, seqb4 = self.model(seqb)
        # p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        # p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqb4).unsqueeze(1)
        # p_a = torch.cat((p_a2,p_a3,p_a4),3)
        # torch.Size([1, 1, 84, 400])
        # torch.Size([1, 194, 64])
        # torch.Size([1, 94, 128])
        # torch.Size([1, 44, 256])
        # torch.Size([1, 38, 512])
        # VGG_out0 = self.VGG_Conv1(p_a1)
        # VGG_out0 = VGG_out0['x5'].view(VGG_out0['x5'].shape[0], -1)

        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat((VGG_out1,VGG_out2, VGG_out3), 1)
        samil = F.softmax(self.fc(VGG_out),dim=1)

        return samil[:,1].unsqueeze(1)

    def get_fixed_out(self, seqa):
        seqa1, seqa2, seqa3, seqa4 = self.model(seqa)
        seqa2, seqa3, seqa4 = self.adp_max_pool(seqa2).squeeze(1), \
                              self.adp_max_pool(seqa3).squeeze(1), \
                              self.adp_max_pool(seqa4).squeeze(1)
        seqa = torch.cat((seqa2, seqa3, seqa4), 1)
        seqa = self.fc1(seqa)
        return seqa
    def forward(self, seqa, seqp, seqn):
        seqa1, seqa2, seqa3, seqa4 = self.model(seqa)
        seqp1, seqp2, seqp3, seqp4 = self.model(seqp)
        seqn1, seqn2, seqn3, seqn4 = self.model(seqn)

        p_a2 = self.metric(seqa2, seqp2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqp3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqp4).unsqueeze(1)

        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat((VGG_out1, VGG_out2, VGG_out3), 1)
        p_ap = self.fc(VGG_out)

        p_b2 = self.metric(seqa2, seqn2).unsqueeze(1)
        p_b3 = self.metric(seqa3, seqn3).unsqueeze(1)
        p_b4 = self.metric(seqa4, seqn4).unsqueeze(1)
        VGG_out1 = self.VGG_Conv1(p_b2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_b3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_b4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat((VGG_out1, VGG_out2, VGG_out3), 1)
        p_an = self.fc(VGG_out)
        seqa2, seqa3, seqa4 = self.adp_max_pool(seqa2).squeeze(1), \
                              self.adp_max_pool(seqa3).squeeze(1), \
                              self.adp_max_pool(seqa4).squeeze(1)
        seqa = torch.cat((seqa2, seqa3, seqa4), 1)
        seqp2, seqp3, seqp4 = self.adp_max_pool(seqp2).squeeze(1), \
                              self.adp_max_pool(seqp3).squeeze(1), \
                              self.adp_max_pool(seqp4).squeeze(1)
        seqp = torch.cat((seqp2, seqp3, seqp4), 1)
        seqn2, seqn3, seqn4 = self.adp_max_pool(seqn2).squeeze(1), \
                              self.adp_max_pool(seqn3).squeeze(1), \
                              self.adp_max_pool(seqn4).squeeze(1)
        seqn = torch.cat((seqn2, seqn3, seqn4), 1)
        seqa ,seqp, seqn = self.fc1(seqa), self.fc1(seqp), self.fc1(seqn)
        seqa_d,seqp_d,seqn_d=self.dropout(seqa),self.dropout(seqp),self.dropout(seqn)
        la, lp, ln = self.fc2(seqa_d), self.fc2(seqp_d), self.fc2(seqn_d)
        return torch.cat((p_ap, p_an), dim=0),seqa,seqp,seqn,la,lp,ln
class SMCNN_5(BasicModule):
    """
    55000it [75:34:19,  4.73s/it]train_loss: 0.007741250745826789
Youtube350:
                         0.9561629266311203 0.1928 1.876
CoverSong80:, 513.07it/s]
                         0.902245744393522 0.0925 4.0875
SH100K:
                         0.7396470873452758 0.49920356499478524 54.79321133971745
*****************BEST*****************
model name 0819_01:03:39.pth
    """
    def __init__(self ):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation_SPP()

        self.VGG_Conv1 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')

        self.fc = nn.Linear(24064, 2)
        self.fc2 = nn.Linear(2048,512)
        self.fc3 = nn.Linear(512,10000)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqp.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return d_ap_s if debug == False else d_ap_s
    def get_fixed_out(self,seqa):
        _, _, _, _,_,x6 = self.model(seqa)
        x6 = self.fc2(x6)
        return x6

    def multi_compute_s(self, seqa, seqb):
        _, seqa2, seqa3, seqa4,seqa5,_  = self.model(seqa)
        _, seqb2, seqb3, seqb4,seqb5,_ = self.model(seqb)
        # p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        # p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqb4).unsqueeze(1)
        # p_a5 = self.metric(seqa5, seqb5).unsqueeze(1)
        # p_a = torch.cat((p_a2,p_a3,p_a4),3)
        # torch.Size([1, 1, 84, 400])
        # torch.Size([1, 194, 64])
        # torch.Size([1, 94, 128])
        # torch.Size([1, 44, 256])
        # torch.Size([1, 38, 512])
        # VGG_out0 = self.VGG_Conv1(p_a1)
        # VGG_out0 = VGG_out0['x5'].view(VGG_out0['x4'].shape[0], -1)
        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        # VGG_out4 = self.VGG_Conv1(p_a5)
        # VGG_out4 = VGG_out4['x2'].view(VGG_out4['x2'].shape[0], -1)
        VGG_out = torch.cat((VGG_out1,VGG_out2, VGG_out3), 1)
        samil = F.softmax(self.fc(VGG_out),dim=1)

        return samil[:,1].unsqueeze(1)

    def forward(self, seqa, seqb, seqn):
        _, seqa2, seqa3, seqa4,seqa5,xa6 = self.model(seqa)

        _, seqb2, seqb3, seqb4,seqb5,xb6 = self.model(seqb)
        # p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqb4).unsqueeze(1)
        # p_a5 = self.metric(seqa5, seqb5).unsqueeze(1)
        # VGG_out0 = self.VGG_Conv1(p_a1)
        # VGG_out0 = VGG_out0['x5'].view(VGG_out0['x4'].shape[0], -1)
        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        # VGG_out4 = self.VGG_Conv1(p_a5)
        # VGG_out4 = VGG_out4['x2'].view(VGG_out4['x2'].shape[0], -1)
        VGG_out = torch.cat(( VGG_out1, VGG_out2, VGG_out3), 1)
        p_ap= self.fc(VGG_out)


        _, seqn2, seqn3, seqn4,seqn5,xn6 = self.model(seqn)
        # n_a1 = self.metric(seqa1, seqn1).unsqueeze(1)
        n_a2 = self.metric(seqa2, seqn2).unsqueeze(1)
        n_a3 = self.metric(seqa3, seqn3).unsqueeze(1)
        n_a4 = self.metric(seqa4, seqn4).unsqueeze(1)
        # n_a5 = self.metric(seqa5, seqn5).unsqueeze(1)
        # VGG_out0 = self.VGG_Conv1(n_a1)
        # VGG_out0 = VGG_out0['x5'].view(VGG_out0['x4'].shape[0], -1)
        VGG_out1 = self.VGG_Conv1(n_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(n_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(n_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        # VGG_out4 = self.VGG_Conv1(n_a5)
        # VGG_out4 = VGG_out4['x2'].view(VGG_out4['x2'].shape[0], -1)
        VGG_out = torch.cat(( VGG_out1, VGG_out2, VGG_out3), 1)
        p_an = self.fc(VGG_out)

        xa6,xb6,xn6= self.fc2(xa6),self.fc2(xb6),self.fc2(xn6)
        pca,pcb,pcn = self.fc3(xa6),self.fc3(xb6),self.fc3(xn6)
        return torch.cat((p_ap, p_an), dim=0),xa6,xb6,xn6, pca,pcb,pcn
if __name__=='__main__':
    y = SMCNN_2().cuda()
    z = y(torch.randn([1,1,84,400]).cuda(),torch.randn([1,1,84,400]).cuda(),torch.randn([1,1,84,400]).cuda())
    print(z[4].shape)