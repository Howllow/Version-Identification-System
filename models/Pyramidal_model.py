import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Function
from collections import OrderedDict
import math
import numpy as np

import matplotlib.pyplot as plt
from numba import jit
from .NonLocalBlock1D import *
from .FCN import *
from .TCN import *
from torch.nn import init
from .BasicModule import BasicModule
from .position_encode import *


class Basic_dilation_pyramid_model(BasicModule):
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
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool0', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
        ]))

        """
        feature1 最终输出每个元素的感受野为：(14 * 0.4644)s
        """
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
        """
        feature2 最终输出每个元素的感受野为：(40 * 0.4644) s
        """
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
        """
        feature3 最终输出每个元素的感受野为：(92 * 0.4644) s
        """
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
        """
        feature3 最终输出每个元素的感受野为：(196 * 0.4644) s
        """
        self.features5 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, 1)))
        ]))
        """
        feature3 最终输出每个元素的感受野为：(202 * 0.4644) s
        """
        self.GRU_2 = nn.GRU(64,64)
        self.GRU_3 = nn.GRU(128, 128)
    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)

        x1 = nn.AdaptiveMaxPool2d((1, None))(x1).squeeze(dim=2).permute(2, 0, 1)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2).permute(2, 0, 1)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2).permute(2, 0, 1)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2).permute(2, 0, 1)
        x5 = x5.squeeze(dim=2).permute(0, 2, 1)
        _,x2 = self.GRU_2(x2)
        _,x3 = self.GRU_3(x3)
        x = torch.cat((x2.permute(1, 0, 2),x3.permute(1, 0, 2),x5),2)
        return x.squeeze(1)
class Basic_dilation_pyramid_model_2(BasicModule):
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
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool0', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
        ]))

        """
        feature1 最终输出每个元素的感受野为：(14 * 0.4644)s
        """
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
        """
        feature2 最终输出每个元素的感受野为：(40 * 0.4644) s
        """
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
        """
        feature3 最终输出每个元素的感受野为：(92 * 0.4644) s
        """
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
        """
        feature3 最终输出每个元素的感受野为：(196 * 0.4644) s
        """
        self.features5 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, 1)))
        ]))
        """
        feature3 最终输出每个元素的感受野为：(202 * 0.4644) s 
        """
        self.GRU_2 = nn.GRU(64 , 128 ,2)
        self.GRU_3 = nn.GRU(128, 256,2)

    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)

        x1 = nn.AdaptiveMaxPool2d((1, None))(x1).squeeze(dim=2).permute(2, 0, 1)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2).permute(2, 0, 1)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2).permute(2, 0, 1)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2).permute(2, 0, 1)
        x5 = x5.squeeze(dim=2).permute(0, 2, 1)

        _,x2 = self.GRU_2(x2)
        _,x3 = self.GRU_3(x3)
        x2,x3 = x2[1].unsqueeze(0),x3[1].unsqueeze(0)

        x = torch.cat((x2.permute(1, 0, 2),x3.permute(1, 0, 2),x5),2)

        return x.squeeze(1)
class NeuralDTW_py1(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model()
        self.feat_bn = nn.BatchNorm1d(704)
        self.Line = torch.nn.Linear(1,1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(704,500)
        self.f2 = torch.nn.Linear(500,10000)
        self.drop=torch.nn.Dropout(0.3)
    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa,seqb = self.f1(seqa),self.f1(seqb)
        smail = torch.sigmoid( self.Line(self.metric(seqa,seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa,seqp,seqn = self.f1(seqa),self.f1(seqp),self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa,seqp))),torch.sigmoid( self.Line(self.metric(seqa,seqn)))
        seqa, seqp, seqn = self.drop(seqa), self.drop(seqp), self.drop(seqn)
        la,lp,ln = self.f2(seqa),self.f2(seqp),self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0),la,lp,ln
class NeuralDTW_py2(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model_2()
        self.feat_bn = nn.BatchNorm1d(704+192)
        self.Line = torch.nn.Linear(1,1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(704+192,500)
        self.f2 = torch.nn.Linear(500,10000)
        self.drop=torch.nn.Dropout(0.3)
    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa,seqb = self.f1(seqa),self.f1(seqb)
        smail = torch.sigmoid( self.Line(self.metric(seqa,seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa,seqp,seqn = self.f1(seqa),self.f1(seqp),self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa,seqp))),torch.sigmoid( self.Line(self.metric(seqa,seqn)))
        seqa, seqp, seqn = self.drop(seqa), self.drop(seqp), self.drop(seqn)
        la,lp,ln = self.f2(seqa),self.f2(seqp),self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0),la,lp,ln
class Basic_dilation_pyramid_model_3(BasicModule):
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
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        feature2 最终输出每个元素的感受野为：(22 * 0.4644) s
        """
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

            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('maxpool4', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
            ('conv10', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU(inplace=True)),
            ('conv11', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm11', nn.BatchNorm2d(512)),
            ('relu11', nn.ReLU(inplace=True)),
        ]))

        self.branch_2 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),

            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),

            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
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
        self.branch_3 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
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
        self.branch_4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
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
        self.GRU= nn.GRU(512 , 512)
        self.relu = torch.nn.ReLU()
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.nolocalATT_2 = NonLocalBlock1D(512, 256)
        self.nolocalATT_3 = NonLocalBlock1D(512, 256)
        self.nolocalATT_4 = NonLocalBlock1D(512, 256)
        # self.feat_bn = nn.BatchNorm1d(1024)
        # self.feat1 = nn.Conv2d(1, 256, kernel_size=(3, 512), stride=1, dilation=(1, 1), padding=(1, 0), bias=False)
        # self.feat2 = nn.Conv2d(1, 128, kernel_size=(3, 512), stride=1, dilation=(2, 1), padding=(2, 0), bias=False)
        # self.feat3 = nn.Conv2d(1, 128, kernel_size=(3, 512), stride=1, dilation=(3, 1), padding=(3, 0), bias=False)
        # init.normal_(self.feat1.weight, std=0.001)
        # init.normal_(self.feat2.weight, std=0.001)
        # init.normal_(self.feat3.weight, std=0.001)
        self.conv_extr = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),

            ('conv5', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm7', nn.BatchNorm2d(512)),
            ('relu7', nn.ReLU(inplace=True)),
            ('conv8', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True))
        ]))
    def forward(self, x):
        x2 = self.features1(x)  # [N, 512, 1, 86]
        x3 = self.features3(x2)
        x3 = self.GAP(x3) + self.GMP(x3)
        x3 = x3.squeeze(dim=2).squeeze(2)
        branch_2 = self.branch_2(x2).squeeze(2)
        branch_3 = self.branch_3(x2).squeeze(2)
        branch_4 = self.branch_4(x2).squeeze(2)
        branch_2 = self.nolocalATT_2(branch_2).mean(2)
        branch_3 = self.nolocalATT_3(branch_3).mean(2)
        branch_4 = self.nolocalATT_4(branch_4).mean(2)

        # x2 = x2.unsqueeze(dim=1)
        # x2_1 = self.feat1(seq2).squeeze(dim=3)
        # x2_2 = self.feat2(seq2).squeeze(dim=3)
        # x2_3 = self.feat3(seq2).squeeze(dim=3)
        # x2 = torch.cat((x2.squeeze(1).permute(0, 2, 1), x2_1, x2_2, x2_3), 1)
        # x2 = self.nolocalATT(x2).mean(dim=2)
        # x2 = self.feat_bn(x2)
        # x2 = torch.cat([x2,x2_],1)
        # x2 = self.relu(x2)
        x = torch.cat((x3,branch_2,branch_3,branch_4),1)
        return x
class Basic_dilation_pyramid_model_4(BasicModule):
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
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool0', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
        ]))

        """
        feature1 最终输出每个元素的感受野为：(14 * 0.4644)s
        """
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
        """
        feature2 最终输出每个元素的感受野为：(40 * 0.4644) s
        """
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
        """
        feature3 最终输出每个元素的感受野为：(92 * 0.4644) s
        """
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
        """
        feature3 最终输出每个元素的感受野为：(196 * 0.4644) s
        """
        self.features5 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, 1)))
        ]))

        """
        feature3 最终输出每个元素的感受野为：(202 * 0.4644) s
        """
        self.GRU= nn.GRU(256 , 256)
        self.relu = torch.nn.ReLU()
        self.nolocalATT = NonLocalBlock1D(1024, 256)
        self.feat_bn = nn.BatchNorm1d(1024)
        self.feat1 = nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1, dilation=(1, 1), padding=(1, 0), bias=False)
        self.feat2 = nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1, dilation=(2, 1), padding=(2, 0), bias=False)
        self.feat3 = nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1, dilation=(3, 1), padding=(3, 0), bias=False)
        init.normal_(self.feat1.weight, std=0.001)
        init.normal_(self.feat2.weight, std=0.001)
        init.normal_(self.feat3.weight, std=0.001)
        self.conv_extr = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),

            ('conv5', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True))
        ]))
    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        x2 = self.conv_extr(x2)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2).permute(2, 0, 1)
        _, x2_ = self.GRU(x2)
        x2_ = x2_.permute(1,0,2).squeeze(1)
        x2 = x2.permute(1,0,2).unsqueeze(dim=1)
        x2_1 = self.feat1(x2).squeeze(dim=3)
        x2_2 = self.feat2(x2).squeeze(dim=3)
        x2_3 = self.feat3(x2).squeeze(dim=3)
        x2 = torch.cat((x2.squeeze(1).permute(0, 2, 1), x2_1, x2_2, x2_3), 1)

        x2 = self.nolocalATT(x2).mean(dim=2)
        x2 = self.feat_bn(x2)

        x2 = torch.cat([x2,x2_],1)
        x2 = self.relu(x2)
        x5 = x5.squeeze(dim=2).squeeze(2)
        x = torch.cat((x2,x5),1)
        return x
class NeuralDTW_py3(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model_3()
        self.feat_bn = nn.BatchNorm1d(1024)
        self.Line = torch.nn.Linear(1,1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(1024,512)
        self.f2 = torch.nn.Linear(512,10000)
        self.drop=torch.nn.Dropout(0.3)
    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa,seqb = self.f1(seqa),self.f1(seqb)
        smail = torch.sigmoid( self.Line(self.metric(seqa,seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa,seqp,seqn = self.f1(seqa),self.f1(seqp),self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa,seqp))),torch.sigmoid( self.Line(self.metric(seqa,seqn)))
        seqa, seqp, seqn = self.drop(seqa), self.drop(seqp), self.drop(seqn)
        la,lp,ln = self.f2(seqa),self.f2(seqp),self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0),la,lp,ln
class NeuralDTW_py4(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model_4()
        self.feat_bn = nn.BatchNorm1d(1792)
        self.Line = torch.nn.Linear(1,1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(1792,512)
        self.f2 = torch.nn.Linear(512,10000)
        self.drop=torch.nn.Dropout(0.3)
    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa,seqb = self.f1(seqa),self.f1(seqb)
        smail = torch.sigmoid( self.Line(self.metric(seqa,seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa,seqp,seqn = self.f1(seqa),self.f1(seqp),self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa,seqp))),torch.sigmoid( self.Line(self.metric(seqa,seqn)))
        seqa, seqp, seqn = self.drop(seqa), self.drop(seqp), self.drop(seqn)
        la,lp,ln = self.f2(seqa),self.f2(seqp),self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0),la,lp,ln

class Basic_dilation_pyramid_model_5(BasicModule):
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
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool0', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
        ]))

        """
        feature1 最终输出每个元素的感受野为：(14 * 0.4644)s
        """
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
        """
        feature2 最终输出每个元素的感受野为：(40 * 0.4644) s
        """
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
        """
        feature3 最终输出每个元素的感受野为：(92 * 0.4644) s
        """
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
        """
        feature3 最终输出每个元素的感受野为：(196 * 0.4644) s
        """
        self.features5 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, 1)))
        ]))

        """
        feature3 最终输出每个元素的感受野为：(202 * 0.4644) s
        """
        self.GRU= nn.GRU(512 , 512)
        self.relu = torch.nn.ReLU()
        # self.nolocalATT = NonLocalBlock1D(1024, 256)
        # self.feat_bn = nn.BatchNorm1d(1024)
        # self.feat1 = nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1, dilation=(1, 1), padding=(1, 0), bias=False)
        # self.feat2 = nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1, dilation=(2, 1), padding=(2, 0), bias=False)
        # self.feat3 = nn.Conv2d(1, 256, kernel_size=(3, 256), stride=1, dilation=(3, 1), padding=(3, 0), bias=False)
        # init.normal_(self.feat1.weight, std=0.001)
        # init.normal_(self.feat2.weight, std=0.001)
        # init.normal_(self.feat3.weight, std=0.001)
        self.conv_extr_1 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),

            ('conv5', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm7', nn.BatchNorm2d(512)),
            ('relu7', nn.ReLU(inplace=True)),
            ('conv8', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)

        # x2 = x2.unsqueeze(dim=1)
        # x2_1 = self.feat1(seq2).squeeze(dim=3)
        # x2_2 = self.feat2(seq2).squeeze(dim=3)
        # x2_3 = self.feat3(seq2).squeeze(dim=3)
        # x2 = torch.cat((x2.squeeze(1).permute(0, 2, 1), x2_1, x2_2, x2_3), 1)
        # x2 = self.nolocalATT(x2).mean(dim=2)
        # x2 = self.feat_bn(x2)
        # x2 = torch.cat([x2,x2_],1)
        # x2 = self.relu(x2)


        x5 = x5.squeeze(dim=2).permute(0, 2, 1)
        x2 = self.conv_extr(x2)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2).permute(2, 0, 1)
        _,x2 = self.GRU(x2)
        x = torch.cat((x2.permute(1, 0, 2),x5),2)
        return x.squeeze(1)
class NeuralDTW_py5(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model_3()
        self.feat_bn = nn.BatchNorm1d(2048)
        self.Line = torch.nn.Linear(1,1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(2048,512)
        self.f2 = torch.nn.Linear(512,10000)
        self.drop=torch.nn.Dropout(0.5)
    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa,seqb = self.f1(seqa),self.f1(seqb)
        smail = torch.sigmoid( self.Line(self.metric(seqa,seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa,seqp,seqn = self.f1(seqa),self.f1(seqp),self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa,seqp))),torch.sigmoid( self.Line(self.metric(seqa,seqn)))
        seqa, seqp, seqn = self.drop(seqa), self.drop(seqp), self.drop(seqn)
        la,lp,ln = self.f2(seqa),self.f2(seqp),self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0),la,lp,ln
class Basic_dilation_pyramid_model_6(BasicModule):
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
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        feature2 最终输出每个元素的感受野为：(22 * 0.4644) s
        """
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

            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('maxpool4', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
            ('conv10', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU(inplace=True)),
            ('conv11', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm11', nn.BatchNorm2d(512)),
            ('relu11', nn.ReLU(inplace=True)),
        ]))

        self.branch_2 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),

            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),

            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
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
        self.branch_3 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
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
        self.branch_4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
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
        self.relu = torch.nn.ReLU()
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.nolocalATT_2 = NonLocalBlock1D_local(512, 128)
        self.nolocalATT_3 = NonLocalBlock1D_local(512, 128)
        self.nolocalATT_4 = NonLocalBlock1D_local(512, 128)
        # self.feat_bn = nn.BatchNorm1d(1024)
        # self.feat1 = nn.Conv2d(1, 256, kernel_size=(3, 512), stride=1, dilation=(1, 1), padding=(1, 0), bias=False)
        # self.feat2 = nn.Conv2d(1, 128, kernel_size=(3, 512), stride=1, dilation=(2, 1), padding=(2, 0), bias=False)
        # self.feat3 = nn.Conv2d(1, 128, kernel_size=(3, 512), stride=1, dilation=(3, 1), padding=(3, 0), bias=False)
        # init.normal_(self.feat1.weight, std=0.001)
        # init.normal_(self.feat2.weight, std=0.001)
        # init.normal_(self.feat3.weight, std=0.001)
    def forward(self, x):
        x2 = self.features1(x)  # [N, 512, 1, 86]
        x3 = self.features3(x2)
        x3 = self.GAP(x3) + self.GMP(x3)
        x3 = x3.squeeze(dim=2).squeeze(2)
        branch_2 = self.branch_2(x2).squeeze(2)
        branch_3 = self.branch_3(x2).squeeze(2)
        branch_4 = self.branch_4(x2).squeeze(2)
        branch_2 = self.nolocalATT_2(branch_2).mean(2)
        branch_3 = self.nolocalATT_3(branch_3).mean(2)
        branch_4 = self.nolocalATT_4(branch_4).mean(2)

        # x2 = x2.unsqueeze(dim=1)
        # x2_1 = self.feat1(seq2).squeeze(dim=3)
        # x2_2 = self.feat2(seq2).squeeze(dim=3)
        # x2_3 = self.feat3(seq2).squeeze(dim=3)
        # x2 = torch.cat((x2.squeeze(1).permute(0, 2, 1), x2_1, x2_2, x2_3), 1)
        # x2 = self.nolocalATT(x2).mean(dim=2)
        # x2 = self.feat_bn(x2)
        # x2 = torch.cat([x2,x2_],1)
        # x2 = self.relu(x2)
        x = torch.cat((x3,branch_2,branch_3,branch_4),1)
        return x

class NeuralDTW_py6(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model_6()
        self.feat_bn = nn.BatchNorm1d(2048)
        self.Line = torch.nn.Linear(1,1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(2048,512)
        self.f2 = torch.nn.Linear(512,10000)
        self.drop=torch.nn.Dropout(0.5)
    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa,seqb = self.f1(seqa),self.f1(seqb)
        smail = torch.sigmoid( self.Line(self.metric(seqa,seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa,seqp,seqn = self.f1(seqa),self.f1(seqp),self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa,seqp))),torch.sigmoid( self.Line(self.metric(seqa,seqn)))
        seqa, seqp, seqn = self.drop(seqa), self.drop(seqp), self.drop(seqn)
        la,lp,ln = self.f2(seqa),self.f2(seqp),self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0),la,lp,ln

class Basic_dilation_pyramid_model_7(BasicModule):
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
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        feature1 最终输出每个元素的感受野为：(22 * 0.4644) s
        """
        self.features2 = nn.Sequential(OrderedDict([
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
        """
        feature2 最终输出每个元素的感受野为：(56 * 0.4644) s
        """
        self.features3 = nn.Sequential(OrderedDict([
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
        """
        feature3 最终输出每个元素的感受野为：(124 * 0.4644) s
        """
        self.features4 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('maxpool4', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        feature4 最终输出每个元素的感受野为：(260 * 0.4644) s
        """
        self.features5 = nn.Sequential(OrderedDict([
            ('conv10', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU(inplace=True)),
            ('conv11', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm11', nn.BatchNorm2d(512)),
            ('relu11', nn.ReLU(inplace=True)),
        ]))
        """
        feature4 最终输出每个元素的感受野为：(266 * 0.4644) s
        """
        self.relu = torch.nn.ReLU()
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.nolocalATT_2 = NonLocalBlock1D_local(128, 128)
        self.nolocalATT_3 = NonLocalBlock1D_local(256, 128)
        self.nolocalATT_4 = NonLocalBlock1D_local(512, 128)

    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        x5 = self.GAP(x5) + self.GMP(x5)
        x5= x5.squeeze(2).squeeze(2)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2).squeeze(2)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2).squeeze(2)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2).squeeze(2)

        branch_2 = self.nolocalATT_2(x2).mean(2)
        branch_3 = self.nolocalATT_3(x3).mean(2)
        branch_4 = self.nolocalATT_4(x4).mean(2)
        x = torch.cat((x5,branch_2,branch_3,branch_4),1)
        return x
class NeuralDTW_py7(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model_7()

        self.Line = torch.nn.Linear(1,1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(1408,512)
        self.f2 = torch.nn.Linear(512,10000)
        self.drop=torch.nn.Dropout(0.7)
    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa,seqb = self.f1(seqa),self.f1(seqb)
        smail = torch.sigmoid( self.Line(self.metric(seqa,seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa,seqp,seqn = self.f1(seqa),self.f1(seqp),self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa,seqp))),torch.sigmoid( self.Line(self.metric(seqa,seqn)))
        seqa, seqp, seqn = self.drop(seqa), self.drop(seqp), self.drop(seqn)
        la,lp,ln = self.f2(seqa),self.f2(seqp),self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0),la,lp,ln


class Basic_dilation_pyramid_model_8(BasicModule):
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
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        feature1 最终输出每个元素的感受野为：(22 * 0.4644) s
        """
        self.features2 = nn.Sequential(OrderedDict([
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
        """
        feature2 最终输出每个元素的感受野为：(56 * 0.4644) s
        """
        self.features3 = nn.Sequential(OrderedDict([
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
        """
        feature3 最终输出每个元素的感受野为：(124 * 0.4644) s
        """
        self.features4 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('maxpool4', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        feature4 最终输出每个元素的感受野为：(260 * 0.4644) s
        """
        self.features5 = nn.Sequential(OrderedDict([
            ('conv10', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                 dilation=(1, 1), bias=False)),
            ('norm10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU(inplace=True)),
            ('conv11', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                 dilation=(1, 2), bias=False)),
            ('norm11', nn.BatchNorm2d(512)),
            ('relu11', nn.ReLU(inplace=True)),
        ]))
        """
        feature4 最终输出每个元素的感受野为：(266 * 0.4644) s
        """
        self.relu = torch.nn.ReLU()
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.nolocalATT_2 = NonLocalBlock1D_local_global_2(128, 64,512)
        self.nolocalATT_3 = NonLocalBlock1D_local_global_2(256, 64,512)
        self.nolocalATT_4 = NonLocalBlock1D_local_global_2(512, 128,512)

    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        x5 = self.GAP(x5) + self.GMP(x5)
        x5 = x5.squeeze(2).squeeze(2)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2).squeeze(2)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2).squeeze(2)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2).squeeze(2)

        branch_2 = self.nolocalATT_2(x2,x5).mean(2)
        branch_3 = self.nolocalATT_3(x3,x5).mean(2)
        branch_4 = self.nolocalATT_4(x4,x5).mean(2)
        x = torch.cat((x5, branch_2, branch_3, branch_4), 1)
        return x

class Basic_dilation_pyramid_model_9(BasicModule):
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
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        feature1 最终输出每个元素的感受野为：(22 * 0.4644) s
        """
        self.features2 = nn.Sequential(OrderedDict([
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
        """
        feature2 最终输出每个元素的感受野为：(56 * 0.4644) s
        """
        self.features3 = nn.Sequential(OrderedDict([
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
        """
        feature3 最终输出每个元素的感受野为：(124 * 0.4644) s
        """
        self.features4 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('maxpool4', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        feature4 最终输出每个元素的感受野为：(260 * 0.4644) s
        """
        self.features5 = nn.Sequential(OrderedDict([
            ('conv10', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                 dilation=(1, 1), bias=False)),
            ('norm10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU(inplace=True)),
            ('conv11', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                 dilation=(1, 2), bias=False)),
            ('norm11', nn.BatchNorm2d(512)),
            ('relu11', nn.ReLU(inplace=True)),
        ]))
        """
        feature4 最终输出每个元素的感受野为：(266 * 0.4644) s
        """
        self.relu = torch.nn.ReLU()
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.nolocalATT_2 = NonLocalBlock1D(128, 64)
        self.nolocalATT_3 = NonLocalBlock1D(256, 64 )
        self.nolocalATT_4 = NonLocalBlock1D(512, 128 )

    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        x5 = self.GAP(x5) + self.GMP(x5)
        x5 = x5.squeeze(2).squeeze(2)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2).squeeze(2)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2).squeeze(2)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2).squeeze(2)

        branch_2 = self.nolocalATT_2(x2).mean(2)
        branch_3 = self.nolocalATT_3(x3).mean(2)
        branch_4 = self.nolocalATT_4(x4).mean(2)
        x = torch.cat((x5, branch_2, branch_3, branch_4), 1)
        return x

class NeuralDTW_py8(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model_8()

        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(1408, 512)
        self.f2 = torch.nn.Linear(512, 10000)
        self.drop = torch.nn.Dropout(0.7)

    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa), self.f1(seqb)
        smail = torch.sigmoid(self.Line(self.metric(seqa, seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa), self.f1(seqp), self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa, seqp))), torch.sigmoid(
            self.Line(self.metric(seqa, seqn)))
        seqa, seqp, seqn = self.drop(seqa), self.drop(seqp), self.drop(seqn)
        la, lp, ln = self.f2(seqa), self.f2(seqp), self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0), la, lp, ln
class NeuralDTW_py9(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model_8()

        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(1408, 512)
        self.f2 = torch.nn.Linear(512, 10000)
        self.drop = torch.nn.Dropout(0.7)

    def metric(self, seqa, seqb, debug=False):

        return torch.cosine_similarity(seqa, seqb, dim=1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa), self.f1(seqb)
        smail = torch.cosine_similarity(seqa, seqb, dim=1)
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa), self.f1(seqp), self.f1(seqn)
        return seqa,seqp,seqn
class NeuralDTW_py10(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model_7()

        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(1408, 512)
        self.f2 = torch.nn.Linear(512, 10000)
        self.drop = torch.nn.Dropout(0.7)

    def metric(self, seqa, seqb, debug=False):

        return torch.cosine_similarity(seqa, seqb, dim=1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa), self.f1(seqb)
        smail = torch.cosine_similarity(seqa, seqb, dim=1)
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa), self.f1(seqp), self.f1(seqn)
        return seqa,seqp,seqn
class NeuralDTW_py11(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model_8()

        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(1408, 512)
        self.f2 = torch.nn.Linear(512, 10000)
        self.drop = torch.nn.Dropout(0.7)

    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa), self.f1(seqb)
        smail = self.metric(seqa, seqb)
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa), self.f1(seqp), self.f1(seqn)
        la, lp, ln = self.f2(seqa), self.f2(seqp), self.f2(seqn)
        return seqa,seqp,seqn, la, lp, ln
class NeuralDTW_py12(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = Basic_dilation_pyramid_model_9()

        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(1408, 512)
        self.f2 = torch.nn.Linear(512, 10000)
        self.drop = torch.nn.Dropout(0.7)

    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa), self.f1(seqb)
        smail = self.metric(seqa, seqb)
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa), self.f1(seqp), self.f1(seqn)
        la, lp, ln = self.f2(seqa), self.f2(seqp), self.f2(seqn)
        return seqa,seqp,seqn, la, lp, ln


class Basic_dilation_pyramid_model_10(BasicModule):
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
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 3), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        """
        out = 13*0.5
        """
        # ('maxpool0',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

        self.features2 = nn.Sequential(OrderedDict([

            ('conv3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 3), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        out = 50*0.5
        """
        self.features3 = nn.Sequential(OrderedDict([

            ('conv6', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(128)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('conv8', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 3), bias=False)),
            ('norm8', nn.BatchNorm2d(256)),
            ('relu8', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        out = 124*0.5
        """
        self.features4 = nn.Sequential(OrderedDict([

            ('conv9', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm9', nn.BatchNorm2d(256)),
            ('relu9', nn.ReLU(inplace=True)),
            ('conv10', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU(inplace=True)),
            ('conv11', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 3), bias=False)),
            ('norm11', nn.BatchNorm2d(512)),
            ('relu11', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, 1)))
        ]))
        self.feat_bn = nn.BatchNorm1d(960)
        self.relu = torch.nn.ReLU()
        self.TPP_ATT_1 = NonLocalBlock1D_new_position(64 ,  64, 3)
        self.TPP_ATT_2 = NonLocalBlock1D_new_position(128,  64, 2)
        self.TPP_ATT_3 = NonLocalBlock1D_new_position(256, 128, 1)
    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]

        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x1 = nn.AdaptiveMaxPool2d((1, None))(x1).squeeze(dim=2)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2)

        """
        torch.Size([2, 64, 388])
        torch.Size([2, 128, 188])
        torch.Size([2, 256, 88])
        torch.Size([256, 88])

        """

        x1 =self.TPP_ATT_1(x1).mean(dim=2)
        x2 = self.TPP_ATT_2(x2).mean(dim=2)
        x3 = self.TPP_ATT_3(x3).mean(dim=2)
        x4 = x4.squeeze(dim=2).squeeze(2)
        x5 = self.feat_bn(torch.cat([x1,x2, x3, x4],1))
        x5 = self.relu(x5)
        return x5
class NeuralDTW_py13(BasicModule):
    def __init__(self, params):
        super().__init__()

        self.model = Basic_dilation_pyramid_model_10( )
        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(960, 320)
        self.f2 = torch.nn.Linear(320, 10000)

    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa), self.f1(seqb)
        smail = torch.sigmoid(self.Line(self.metric(seqa, seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa), self.f1(seqp), self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa, seqp))), torch.sigmoid(
            self.Line(self.metric(seqa, seqn)))
        la, lp, ln = self.f2(seqa), self.f2(seqp), self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0), la, lp, ln

class Basic_dilation_pyramid_model_11(BasicModule):
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
            ('pool0', nn.AdaptiveMaxPool2d((1, 1)))
        ]))
        self.TPP_ATT_2 = NonLocalBlock1D_new_position(64, 64, 3)
        self.TPP_ATT_3 = NonLocalBlock1D_new_position(128, 64, 2)
        self.TPP_ATT_4 = NonLocalBlock1D_new_position(256, 128, 1)
        self.feat_bn = nn.BatchNorm1d(960)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x2).squeeze(dim=2)
        x2 = self.TPP_ATT_2(x2).mean(2)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x3).squeeze(dim=2)
        x3 = self.TPP_ATT_3(x3).mean(2)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x4).squeeze(dim=2)
        x4 = self.TPP_ATT_4(x4).mean(2)
        x5 = x5.squeeze(dim=2).squeeze(dim=2)
        x6 = torch.cat([ x2, x3, x4, x5],1)
        x6 = self.feat_bn(x6)
        x6= self.relu(x6)
        return x6
class NeuralDTW_py14(BasicModule):
    def __init__(self, params):
        super().__init__()

        self.model = Basic_dilation_pyramid_model_11( )
        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(960, 320)
        self.f2 = torch.nn.Linear(320, 10000)

    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa), self.f1(seqb)
        smail = torch.sigmoid(self.Line(self.metric(seqa, seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa), self.f1(seqp), self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa, seqp))), torch.sigmoid(
            self.Line(self.metric(seqa, seqn)))
        la, lp, ln = self.f2(seqa), self.f2(seqp), self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0), la, lp, ln


"""
-------------------------------------------------------------------
"""
class Basic_dilation_pyramid_model_12(BasicModule):
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
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation=(1, 3), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        """
        out = 13*0.5
        """
        # ('maxpool0',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

        self.features2 = nn.Sequential(OrderedDict([

            ('conv3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 3), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        out = 50*0.5
        """
        self.features3 = nn.Sequential(OrderedDict([

            ('conv6', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(128)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('conv8', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 3), bias=False)),
            ('norm8', nn.BatchNorm2d(256)),
            ('relu8', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))),
        ]))
        """
        out = 124*0.5
        """
        self.features4 = nn.Sequential(OrderedDict([

            ('conv9', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation=(1, 1), bias=False)),
            ('norm9', nn.BatchNorm2d(256)),
            ('relu9', nn.ReLU(inplace=True)),
            ('conv10', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 2), bias=False)),
            ('norm10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU(inplace=True)),
            ('conv11', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation=(1, 3), bias=False)),
            ('norm11', nn.BatchNorm2d(512)),
            ('relu11', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, 1)))
        ]))
        self.feat_bn = nn.BatchNorm1d(960)
        self.relu = torch.nn.ReLU()
        self.TPP_ATT_1 = NonLocalBlock1D_new_position_2(64 ,  64, 3)
        self.TPP_ATT_2 = NonLocalBlock1D_new_position_2(128,  64, 2)
        self.TPP_ATT_3 = NonLocalBlock1D_new_position_2(256, 128, 1)
    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]

        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x1 = nn.AdaptiveMaxPool2d((1, None))(x1).squeeze(dim=2)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2)

        """
        torch.Size([2, 64, 388])
        torch.Size([2, 128, 188])
        torch.Size([2, 256, 88])
        torch.Size([256, 88])

        """

        x1 =self.TPP_ATT_1(x1).mean(dim=2)
        x2 = self.TPP_ATT_2(x2).mean(dim=2)
        x3 = self.TPP_ATT_3(x3).mean(dim=2)
        x4 = x4.squeeze(dim=2).squeeze(2)
        x5 = self.feat_bn(torch.cat([x1,x2, x3, x4],1))
        x5 = self.relu(x5)
        return x5
class NeuralDTW_py15(BasicModule):
    def __init__(self, params):
        super().__init__()

        self.model = Basic_dilation_pyramid_model_12( )
        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(960, 320)
        self.f2 = torch.nn.Linear(320, 10000)

    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa), self.f1(seqb)
        smail = torch.sigmoid(self.Line(self.metric(seqa, seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa), self.f1(seqp), self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa, seqp))), torch.sigmoid(
            self.Line(self.metric(seqa, seqn)))
        la, lp, ln = self.f2(seqa), self.f2(seqp), self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0), la, lp, ln

class Basic_dilation_pyramid_model_13(BasicModule):
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
            ('pool0', nn.AdaptiveMaxPool2d((1, 1)))
        ]))
        self.TPP_ATT_2 = NonLocalBlock1D_new_position_2(64, 64, 3)
        self.TPP_ATT_3 = NonLocalBlock1D_new_position_2(128, 64, 2)
        self.TPP_ATT_4 = NonLocalBlock1D_new_position_2(256, 128, 1)
        self.feat_bn = nn.BatchNorm1d(960)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x2).squeeze(dim=2)
        x2 = self.TPP_ATT_2(x2).mean(2)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x3).squeeze(dim=2)
        x3 = self.TPP_ATT_3(x3).mean(2)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x4).squeeze(dim=2)
        x4 = self.TPP_ATT_4(x4).mean(2)
        x5 = x5.squeeze(dim=2).squeeze(dim=2)
        x6 = torch.cat([ x2, x3, x4, x5],1)
        x6 = self.feat_bn(x6)
        x6= self.relu(x6)
        return x6
class NeuralDTW_py16(BasicModule):
    def __init__(self, params):
        super().__init__()

        self.model = Basic_dilation_pyramid_model_13( )
        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(960, 320)
        self.f2 = torch.nn.Linear(320, 10000)

    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa), self.f1(seqb)
        smail = torch.sigmoid(self.Line(self.metric(seqa, seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa), self.f1(seqp), self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa, seqp))), torch.sigmoid(
            self.Line(self.metric(seqa, seqn)))
        la, lp, ln = self.f2(seqa), self.f2(seqp), self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0), la, lp, ln

class Basic_dilation_pyramid_model_14(BasicModule):
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
            ('pool0', nn.AdaptiveMaxPool2d((1, 1)))
        ]))
        self.TPP_ATT_2 = NonLocalBlock1D_new_position_multi_head(64, 64, 3)
        self.TPP_ATT_3 = NonLocalBlock1D_new_position_multi_head(128, 64, 2)
        self.TPP_ATT_4 = NonLocalBlock1D_new_position_multi_head(256, 128, 1)
        self.feat_bn = nn.BatchNorm1d(960)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x2).squeeze(dim=2)
        x2 = self.TPP_ATT_2(x2).mean(2)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x3).squeeze(dim=2)
        x3 = self.TPP_ATT_3(x3).mean(2)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x4).squeeze(dim=2)
        x4 = self.TPP_ATT_4(x4).mean(2)
        x5 = x5.squeeze(dim=2).squeeze(dim=2)
        x6 = torch.cat([ x2, x3, x4, x5],1)
        x6 = self.feat_bn(x6)
        x6= self.relu(x6)
        return x6
class NeuralDTW_py17(BasicModule):
    def __init__(self, params):
        super().__init__()

        self.model = Basic_dilation_pyramid_model_14( )
        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(960, 320)
        self.f2 = torch.nn.Linear(320, 10000)

    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa), self.f1(seqb)
        smail = torch.sigmoid(self.Line(self.metric(seqa, seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa), self.f1(seqp), self.f1(seqn)
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa, seqp))), torch.sigmoid(
            self.Line(self.metric(seqa, seqn)))
        la, lp, ln = self.f2(seqa), self.f2(seqp), self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0), la, lp, ln
class Basic_dilation_pyramid_model_15(BasicModule):
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
            ('pool0', nn.AdaptiveMaxPool2d((1, 1)))
        ]))
        self.TPP_ATT_2 = NonLocalBlock1D_new_position_multi_head_new(64,  64 , 256, 3)
        self.TPP_ATT_3 = NonLocalBlock1D_new_position_multi_head_new(128, 64 , 256, 2)
        self.TPP_ATT_4 = NonLocalBlock1D_new_position_multi_head_new(256, 128, 256, 1)
        self.feat_bn = nn.BatchNorm1d(256*3+512)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x2).squeeze(dim=2)
        x2 = self.TPP_ATT_2(x2).mean(2)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x3).squeeze(dim=2)
        x3 = self.TPP_ATT_3(x3).mean(2)
        x4 = nn.AdaptiveMaxPool2d((1, None))(x4).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x4).squeeze(dim=2)
        x4 = self.TPP_ATT_4(x4).mean(2)
        x5 = x5.squeeze(dim=2).squeeze(dim=2)
        x6 = torch.cat([ x2, x3, x4, x5],1)
        x6 = self.feat_bn(x6)
        x6= self.relu(x6)
        return x2,x3,x4,x5,x6
class NeuralDTW_py18(BasicModule):
    """
    使用position 同时使用多头注意力并且不同尺度最终得到的数据维数相同
    """
    def __init__(self, params):
        super().__init__()

        self.model = Basic_dilation_pyramid_model_15( )
        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(256*3+512, 500)
        self.f2 = torch.nn.Linear(500, 10000)

    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa[4]), self.f1(seqb[4])
        smail = torch.sigmoid(self.Line(self.metric(seqa, seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa[4]), self.f1(seqp[4]), self.f1(seqn[4])
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa, seqp))), torch.sigmoid(
            self.Line(self.metric(seqa, seqn)))
        la, lp, ln = self.f2(seqa), self.f2(seqp), self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0), la, lp, ln
class Basic_dilation_pyramid_model_16(BasicModule):
    """
    不使用position 同时使用多头注意力并且不同尺度最终得到的数据维数相同
    """
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
            ('pool0', nn.AdaptiveMaxPool2d((1, 1)))
        ]))
        self.TPP_ATT_2 = NonLocalBlock1D_multi_head_new(64,  64 , 256, 3)
        self.TPP_ATT_3 = NonLocalBlock1D_multi_head_new(128, 64 , 256, 2)
        self.feat_bn = nn.BatchNorm1d(256*2+512)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x1 = self.features1(x)  # [N, 512, 1, 86]
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        x2 = nn.AdaptiveMaxPool2d((1, None))(x2).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x2).squeeze(dim=2)
        x2 = self.TPP_ATT_2(x2).mean(2)
        x3 = nn.AdaptiveMaxPool2d((1, None))(x3).squeeze(dim=2)+nn.AdaptiveAvgPool2d((1, None))(x3).squeeze(dim=2)
        x3 = self.TPP_ATT_3(x3).mean(2)
        x5 = x5.squeeze(dim=2).squeeze(dim=2)
        x6 = torch.cat([ x2, x3, x5],1)
        x6 = self.feat_bn(x6)
        x6= self.relu(x6)
        return x2,x3,x5,x6
class NeuralDTW_py19(BasicModule):
    def __init__(self, params):
        super().__init__()

        self.model = Basic_dilation_pyramid_model_16( )
        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(256*2+512, 500)
        self.f2 = torch.nn.Linear(500, 10000)

    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        seqa, seqb = self.f1(seqa[3]), self.f1(seqb[3])
        smail = torch.sigmoid(self.Line(self.metric(seqa, seqb)))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)
        seqa, seqp, seqn = self.f1(seqa[3]), self.f1(seqp[3]), self.f1(seqn[3])
        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa, seqp))), torch.sigmoid(
            self.Line(self.metric(seqa, seqn)))
        la, lp, ln = self.f2(seqa), self.f2(seqp), self.f2(seqn)
        return torch.cat((p_ap, p_an), dim=0), la, lp, ln
class NeuralDTW_py19_re(BasicModule):
    def __init__(self, params):
        super().__init__()

        self.model = Basic_dilation_pyramid_model_16( )
        self.relu = torch.nn.ReLU()
        self.f1 = torch.nn.Linear(1024, 300)
        self.f2 = torch.nn.Linear(300, 10000)


    def forward(self, seq ):
        _,_,_,seqa = self.model(seq)
        seqa  = self.f1(seqa)
        la = self.f2(seqa)
        return la,seqa

class NeuralDTW_py20(BasicModule):
    def __init__(self, params):
        super().__init__()

        self.model = Basic_dilation_pyramid_model_16( )
        self.Line = torch.nn.Linear(1, 1)
        self.relu = torch.nn.ReLU()
        self.f2 = torch.nn.Linear(256,10000)
        self.f3 = torch.nn.Linear(256,10000)
        self.f5 = torch.nn.Linear(512,10000)
    def metric(self, seqa, seqb, debug=False):
        smail = torch.sum((seqa - seqb) ** 2, dim=1)
        return smail.unsqueeze(1)

    def multi_compute_s(self, seqa, seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        smail = torch.sigmoid(self.Line(self.metric(seqa[3], seqb[3])))
        return smail

    def forward(self, seqa, seqp, seqn):
        seqa = self.model(seqa)
        seqp = self.model(seqp)
        seqn = self.model(seqn)

        p_ap, p_an = torch.sigmoid(self.Line(self.metric(seqa[3], seqp[3]))), torch.sigmoid(
            self.Line(self.metric(seqa[3], seqn[3])))
        la = [self.f2(seqa[0]), self.f3(seqa[1]),  self.f5(seqa[2])]
        lp = [self.f2(seqp[0]), self.f3(seqp[1]),  self.f5(seqp[2])]
        ln = [self.f2(seqn[0]), self.f3(seqn[1]),  self.f5(seqn[2])]
        return torch.cat((p_ap, p_an), dim=0), la, lp, ln


if __name__ == '__main__':
    x = NeuralDTW_py20(2).cuda()
    y = x(torch.randn([2,1,84,300]).cuda(),torch.randn([2,1,84,300]).cuda(),torch.randn([2,1,84,300]).cuda())
    print(y[0].shape)
    # torch.Size([2, 247, 64])
    # torch.Size([2, 120, 64])
    # torch.Size([2, 57, 128])
    # torch.Size([2, 25, 256])
    # torch.Size([2, 512, 1, 19])
