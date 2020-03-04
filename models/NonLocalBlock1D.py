import torch
from torch import nn
from torch.nn import functional as F
from .position_encode import *

class NonLocalBlock1D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock1D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)       

        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
    def get_att_map(self,x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        return f_div_C
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
class NonLocalBlock1D_position(nn.Module):
    def __init__(self, in_channels, inter_channels=None,temporal_layer_num=2,emb_dim=128,max_len=10,batch_size=24):
        super(NonLocalBlock1D_position, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.z = nn.Conv1d(self.in_channels*(1+temporal_layer_num), self.in_channels, kernel_size=1, stride=1, padding=0)

        self.g = nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.temporal_conv = self._make_conv(num=temporal_layer_num,in_channel = self.in_channels,out_channel=self.in_channels)
        self.pos_encode = PositionalEncoder(emb_dim = emb_dim,max_len =max_len,batch_size = batch_size )
        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
    def _make_conv(self,num,in_channel,out_channel):
        s = []
        if num ==0 :
            return
        for i in range(num):
            s.append(nn.Conv2d(1, out_channels=out_channel , kernel_size=(3, in_channel), stride=1, dilation=(i+1, 1), padding=(i+1, 0), bias=False).cuda())
            init.normal_(s[i].weight, std=0.001)
        return s
    def _get_TP_feature(self,x):
        s = []
        if self.temporal_conv==None:
            return
        x = x.permute(0,2,1).unsqueeze(1)
        for i in range(len(self.temporal_conv)):
            s.append(self.temporal_conv[i](x).squeeze(3).cuda())
        return torch.cat(s,1)
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        pos = self.pos_encode()
        x = x + pos.permute(0,2,1).cuda()
        temp_x = self._get_TP_feature(x.cuda())
        x = torch.cat([x, temp_x], 1)
        x = self.z(x).view(batch_size, self.in_channels, -1)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NonLocalBlock1D_new_position(nn.Module):
    def __init__(self, in_channels, inter_channels=None,temporal_layer_num=2):
        super(NonLocalBlock1D_new_position, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.z = nn.Conv1d(self.in_channels*(1+temporal_layer_num), self.in_channels, kernel_size=1, stride=1, padding=0)

        self.g = nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.temporal_conv = self._make_conv(num=temporal_layer_num,in_channel = self.in_channels,out_channel=self.in_channels)

        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
    def _make_conv(self,num,in_channel,out_channel):
        s = []
        if num ==0 :
            return
        for i in range(num):
            s.append(nn.Conv2d(1, out_channels=out_channel , kernel_size=(3, in_channel), stride=1, dilation=(i+1, 1), padding=(i+1, 0), bias=False).cuda())
            init.normal_(s[i].weight, std=0.001)
        return s
    def _get_TP_feature(self,x):
        s = []
        if self.temporal_conv==None:
            return
        x = x.permute(0,2,1).unsqueeze(1)
        for i in range(len(self.temporal_conv)):
            s.append(self.temporal_conv[i](x).squeeze(3).cuda())
        return torch.cat(s,1)
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        pos =  PositionalEncoder(emb_dim = x.shape[1],max_len =x.shape[2],batch_size = batch_size ).cuda()
        x = x + pos().permute(0,2,1)
        temp_x = self._get_TP_feature(x.cuda())
        x = torch.cat([x, temp_x], 1)
        x = self.z(x).view(batch_size, self.in_channels, -1)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NonLocalBlock1D_new_position_2(nn.Module):
    def __init__(self, in_channels, inter_channels=None,temporal_layer_num=2):
        super(NonLocalBlock1D_new_position_2, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.z = nn.Conv1d(self.in_channels*(1+temporal_layer_num), self.in_channels, kernel_size=1, stride=1, padding=0)

        self.g = nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.temporal_conv = self._make_conv(num=temporal_layer_num,in_channel = self.in_channels,out_channel=self.in_channels)
        self.tranpos_w =  nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
    def _make_conv(self,num,in_channel,out_channel):
        s = []
        if num ==0 :
            return
        for i in range(num):
            s.append(nn.Conv2d(1, out_channels=out_channel , kernel_size=(3, in_channel), stride=1, dilation=(i+1, 1), padding=(i+1, 0), bias=False).cuda())
            init.normal_(s[i].weight, std=0.001)
        return s
    def _get_TP_feature(self,x):
        s = []
        if self.temporal_conv==None:
            return
        x = x.permute(0,2,1).unsqueeze(1)
        for i in range(len(self.temporal_conv)):
            s.append(self.temporal_conv[i](x).squeeze(3).cuda())
        return torch.cat(s,1)
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        pos =  PositionalEncoder_2(emb_dim = x.shape[1],max_len =x.shape[2],batch_size = batch_size )
        x = x + pos().permute(0,2,1).cuda()
        x = self.relu(self.tranpos_w(x))
        temp_x = self._get_TP_feature(x.cuda())
        x = torch.cat([x, temp_x], 1)
        x = self.z(x).view(batch_size, self.in_channels, -1)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

from torch.nn import init
class NonLocalBlock1D_local(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock1D_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, self.inter_channels , kernel_size=1, stride=1, padding=0)
        self.feat1 = nn.Conv2d(1, self.inter_channels , kernel_size=(3, self.inter_channels ), stride=1, dilation=(1, 1), padding=(1, 0), bias=False)
        self.feat2 = nn.Conv2d(1, self.inter_channels , kernel_size=(3, self.inter_channels ), stride=1, dilation=(2, 1), padding=(2, 0), bias=False)
        self.feat3 = nn.Conv2d(1, self.inter_channels , kernel_size=(3, self.inter_channels ), stride=1, dilation=(3, 1), padding=(3, 0), bias=False)

        self.v_w = nn.Conv1d(self.inter_channels*4, self.inter_channels, kernel_size=1, stride=1, padding=0)
        init.normal_(self.feat1.weight, std=0.001)
        init.normal_(self.feat2.weight, std=0.001)
        # init.normal_(self.feat3.weight, std=0.001)
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        v_x_1 = self.feat1(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        v_x_2 = self.feat2(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        v_x_3 = self.feat3(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        phi_x = torch.cat([phi_x,v_x_1,v_x_2,v_x_3],1)

        phi_x = self.v_w(phi_x)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class NonLocalBlock1D_local_new(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock1D_local_new, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, self.inter_channels , kernel_size=1, stride=1, padding=0)
        self.feat1 = nn.Conv2d(1, self.inter_channels , kernel_size=(3, self.inter_channels ), stride=1, dilation=(1, 1), padding=(1, 0), bias=False)
        self.feat2 = nn.Conv2d(1, self.inter_channels , kernel_size=(3, self.inter_channels ), stride=1, dilation=(2, 1), padding=(2, 0), bias=False)
        self.feat3 = nn.Conv2d(1, self.inter_channels , kernel_size=(3, self.inter_channels ), stride=1, dilation=(3, 1), padding=(3, 0), bias=False)

        self.v_w = nn.Conv1d(self.inter_channels*4, self.inter_channels, kernel_size=1, stride=1, padding=0)
        init.normal_(self.feat1.weight, std=0.001)
        init.normal_(self.feat2.weight, std=0.001)
        # init.normal_(self.feat3.weight, std=0.001)
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)


        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        v_x_1 = self.feat1(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        v_x_2 = self.feat2(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        v_x_3 = self.feat3(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        phi_x = torch.cat([phi_x,v_x_1,v_x_2,v_x_3],1)

        phi_x = self.v_w(phi_x)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class NonLocalBlock1D_local_global(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock1D_local_global, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.feat1 = nn.Conv2d(1, self.inter_channels, kernel_size=(3, self.inter_channels), stride=1, dilation=(1, 1),
                               padding=(1, 0), bias=False)
        self.feat2 = nn.Conv2d(1, self.inter_channels, kernel_size=(3, self.inter_channels), stride=1, dilation=(2, 1),
                               padding=(2, 0), bias=False)
        self.feat3 = nn.Conv2d(1, self.inter_channels, kernel_size=(3, self.inter_channels), stride=1, dilation=(3, 1),
                               padding=(3, 0), bias=False)

        self.v_w = nn.Conv1d(self.inter_channels * 4, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta_w = nn.Conv1d(self.inter_channels * 4, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta_u = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi_u = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        init.normal_(self.feat1.weight, std=0.001)
        init.normal_(self.feat2.weight, std=0.001)
        # init.normal_(self.feat3.weight, std=0.001)

    def forward(self, x,global_vec):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        global_vec = global_vec.unsqueeze(2).repeat(1,1,x.size()[2])
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x_1 = self.feat1(theta_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        theta_x_2 = self.feat2(theta_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        theta_x_3 = self.feat3(theta_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        theta_x = torch.cat([theta_x, theta_x_1, theta_x_2, theta_x_3], 1)
        theta_x = self.theta_w(theta_x)
        theta_x = theta_x.permute(0, 2, 1)


        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        v_x_1 = self.feat1(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        v_x_2 = self.feat2(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        v_x_3 = self.feat3(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        phi_x = torch.cat([phi_x, v_x_1, v_x_2, v_x_3], 1)
        phi_x = self.v_w(phi_x)
        theta_x_u = self.theta_u(global_vec).view(batch_size, self.inter_channels, -1)
        theta_x_u= theta_x_u.permute(0, 2, 1)
        phi_x_u = self.phi_u(global_vec).view(batch_size, self.inter_channels, -1)
        theta_x = 0.7*theta_x+0.3*theta_x_u
        phi_x = phi_x*0.7+0.3*phi_x_u

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z
class NonLocalBlock1D_local_global_2(nn.Module):
    def __init__(self, in_channels, inter_channels=None,global_channelsize=None):
        super(NonLocalBlock1D_local_global_2, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.feat1 = nn.Conv2d(1, self.inter_channels, kernel_size=(3, self.inter_channels), stride=1, dilation=(1, 1),
                               padding=(1, 0), bias=False)
        self.feat2 = nn.Conv2d(1, self.inter_channels, kernel_size=(3, self.inter_channels), stride=1, dilation=(2, 1),
                               padding=(2, 0), bias=False)
        self.feat3 = nn.Conv2d(1, self.inter_channels, kernel_size=(3, self.inter_channels), stride=1, dilation=(3, 1),
                               padding=(3, 0), bias=False)
        self.feat1_2 = nn.Conv2d(1, self.inter_channels, kernel_size=(3, self.inter_channels), stride=1, dilation=(1, 1),
                               padding=(1, 0), bias=False)
        self.feat2_2 = nn.Conv2d(1, self.inter_channels, kernel_size=(3, self.inter_channels), stride=1, dilation=(2, 1),
                               padding=(2, 0), bias=False)
        self.feat3_2 = nn.Conv2d(1, self.inter_channels, kernel_size=(3, self.inter_channels), stride=1, dilation=(3, 1),
                               padding=(3, 0), bias=False)
        self.v_w = nn.Conv1d(self.inter_channels * 4, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta_w = nn.Conv1d(self.inter_channels * 4, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta_u = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi_u = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        init.normal_(self.feat1.weight, std=0.001)
        init.normal_(self.feat2.weight, std=0.001)
        # init.normal_(self.feat3.weight, std=0.001)
        self.trans_g =  nn.Conv1d(global_channelsize, in_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x,global_vec):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        global_vec = global_vec.unsqueeze(2).repeat(1,1,x.size()[2])
        global_vec = self.trans_g(global_vec)
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x_1 = self.feat1(theta_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        theta_x_2 = self.feat2(theta_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        theta_x_3 = self.feat3(theta_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        theta_x = torch.cat([theta_x, theta_x_1, theta_x_2, theta_x_3], 1)
        theta_x = self.theta_w(theta_x)
        theta_x = theta_x.permute(0, 2, 1)


        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        v_x_1 = self.feat1_2(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        v_x_2 = self.feat2_2(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        v_x_3 = self.feat3_2(phi_x.permute(0, 2, 1).unsqueeze(1)).squeeze(3)
        phi_x = torch.cat([phi_x, v_x_1, v_x_2, v_x_3], 1)
        phi_x = self.v_w(phi_x)
        theta_x_u = self.theta_u(global_vec).view(batch_size, self.inter_channels, -1)
        theta_x_u= theta_x_u.permute(0, 2, 1)
        phi_x_u = self.phi_u(global_vec).view(batch_size, self.inter_channels, -1)
        theta_x = 0.7*theta_x+0.3*theta_x_u
        phi_x = phi_x*0.7+0.3*phi_x_u

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z
class NonLocalBlock1D_global(nn.Module):
    def __init__(self, in_channels, inter_channels=None,global_channelsize=None):
        super(NonLocalBlock1D_local_global_2, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.theta_w = nn.Conv1d(self.inter_channels * 4, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta_u = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi_u = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        # init.normal_(self.feat3.weight, std=0.001)
        self.trans_g =  nn.Conv1d(global_channelsize, in_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x,global_vec):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        global_vec = global_vec.unsqueeze(2).repeat(1,1,x.size()[2])
        global_vec = self.trans_g(global_vec)
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)


        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        theta_x_u = self.theta_u(global_vec).view(batch_size, self.inter_channels, -1)
        theta_x_u= theta_x_u.permute(0, 2, 1)
        phi_x_u = self.phi_u(global_vec).view(batch_size, self.inter_channels, -1)
        theta_x = 0.7*theta_x+0.3*theta_x_u
        phi_x = phi_x*0.7+0.3*phi_x_u

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z
class NonLocalBlock1D_new_position_multi_head(nn.Module):
    def __init__(self, in_channels, inter_channels=None,temporal_layer_num=2):
        super(NonLocalBlock1D_new_position_multi_head, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.z = nn.Conv1d(self.in_channels*(1+temporal_layer_num), self.in_channels, kernel_size=1, stride=1, padding=0)



        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels*(temporal_layer_num+1), in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.temporal_conv = self._make_conv(num=temporal_layer_num,in_channel = self.in_channels,out_channel=self.in_channels)
        self.temporal_layer_num = temporal_layer_num
        self.tranpos_w =  nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.g = self._make_w(temporal_layer_num+1)
        self.theta = self._make_w(temporal_layer_num+1)
        self.phi = self._make_w(temporal_layer_num+1)
        self.relu = torch.nn.ReLU()
    def _make_w(self,num):
        s=[]
        for i in range(num):
            s.append( nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0).cuda())
        return s
    def _make_conv(self,num,in_channel,out_channel):
        s = []
        if num ==0 :
            return
        for i in range(num):
            s.append(nn.Conv2d(1, out_channels=out_channel , kernel_size=(3, in_channel), stride=1, dilation=(i+1, 1), padding=(i+1, 0), bias=False).cuda())
            init.normal_(s[i].weight, std=0.001)
        return s
    def _get_TP_feature(self,x):
        s = []
        if self.temporal_conv==None:
            return
        x = x.permute(0,2,1).unsqueeze(1)
        for i in range(len(self.temporal_conv)):
            s.append(self.temporal_conv[i](x).squeeze(3).cuda())
        return s
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        pos =  PositionalEncoder_2(emb_dim = x.shape[1],max_len =x.shape[2],batch_size = batch_size )
        x = x + pos().permute(0,2,1).cuda()
        x = self.relu(self.tranpos_w(x))
        multi_att_list=[]
        temp_x = self._get_TP_feature(x)
        temp_x.append(x)
        for i in range(self.temporal_layer_num+1):
            temp_g_x = self.g[i](temp_x[i]).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            temp_theta_x = self.theta[i](temp_x[i]).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            temp_phi_x = self.phi[i](temp_x[i]).view(batch_size, self.inter_channels, -1)
            temp_f = torch.matmul(temp_theta_x, temp_phi_x)
            temp_f = F.softmax(temp_f, dim=-1)
            temp_y = torch.matmul(temp_f, temp_g_x).permute(0, 2, 1).contiguous()
            temp_y = temp_y.view(batch_size, self.inter_channels, *x.size()[2:])
            multi_att_list.append(temp_y)
        y = torch.cat(multi_att_list,1)

        W_y = self.W(y)
        z = W_y + x

        return z
class NonLocalBlock1D_new_position_multi_head_new(nn.Module):
    def __init__(self, in_channels, inter_channels=None,out_channels = 512,temporal_layer_num=2):
        super(NonLocalBlock1D_new_position_multi_head_new, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.z = nn.Conv1d(self.in_channels*(1+temporal_layer_num), self.in_channels, kernel_size=1, stride=1, padding=0)



        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels*(temporal_layer_num+1), in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.temporal_conv = self._make_conv(num=temporal_layer_num,in_channel = self.in_channels,out_channel=self.in_channels)
        self.temporal_layer_num = temporal_layer_num
        self.tranpos_w =  nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.f_x = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels)
        )
        self.g = self._make_w(temporal_layer_num+1)
        self.theta = self._make_w(temporal_layer_num+1)
        self.phi = self._make_w(temporal_layer_num+1)
        self.relu = torch.nn.ReLU()
    def _make_w(self,num):
        s=[]
        for i in range(num):
            s.append( nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0).cuda())
        return s
    def _make_conv(self,num,in_channel,out_channel):
        s = []
        if num ==0 :
            return
        for i in range(num):
            s.append(nn.Conv2d(1, out_channels=out_channel , kernel_size=(3, in_channel), stride=1, dilation=(i+1, 1), padding=(i+1, 0), bias=False).cuda())
            init.normal_(s[i].weight, std=0.001)
        return s
    def _get_TP_feature(self,x):
        s = []
        if self.temporal_conv==None:
            return
        x = x.permute(0,2,1).unsqueeze(1)
        for i in range(len(self.temporal_conv)):
            s.append(self.temporal_conv[i](x).squeeze(3).cuda())
        return s
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        pos =  PositionalEncoder_2(emb_dim = x.shape[1],max_len =x.shape[2],batch_size = batch_size )
        x = x + pos().permute(0,2,1).cuda()
        x = self.relu(self.tranpos_w(x))
        multi_att_list=[]
        temp_x = self._get_TP_feature(x)
        temp_x.append(x)
        for i in range(self.temporal_layer_num+1):
            temp_g_x = self.g[i](temp_x[i]).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            temp_theta_x = self.theta[i](temp_x[i]).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            temp_phi_x = self.phi[i](temp_x[i]).view(batch_size, self.inter_channels, -1)
            temp_f = torch.matmul(temp_theta_x, temp_phi_x)
            temp_f = F.softmax(temp_f, dim=-1)
            temp_y = torch.matmul(temp_f, temp_g_x).permute(0, 2, 1).contiguous()
            temp_y = temp_y.view(batch_size, self.inter_channels, *x.size()[2:])
            multi_att_list.append(temp_y)
        y = torch.cat(multi_att_list,1)

        W_y = self.W(y)
        z = W_y + x
        z = self.f_x(z)
        return z
class NonLocalBlock1D_multi_head_new(nn.Module):
    def __init__(self, in_channels, inter_channels=None,out_channels = 512,temporal_layer_num=2):
        super(NonLocalBlock1D_multi_head_new, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.z = nn.Conv1d(self.in_channels*(1+temporal_layer_num), self.in_channels, kernel_size=1, stride=1, padding=0)



        self.W = nn.Sequential(
            nn.Conv1d(self.inter_channels*(temporal_layer_num+1), in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.temporal_conv = self._make_conv(num=temporal_layer_num,in_channel = self.in_channels,out_channel=self.in_channels)
        self.temporal_layer_num = temporal_layer_num
        self.tranpos_w =  nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.f_x = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels)
        )
        self.g = self._make_w(temporal_layer_num+1)
        self.theta = self._make_w(temporal_layer_num+1)
        self.phi = self._make_w(temporal_layer_num+1)
        self.relu = torch.nn.ReLU()
    def _make_w(self,num):
        s=[]
        for i in range(num):
            s.append( nn.Conv1d(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0).cuda())
        return s
    def _make_conv(self,num,in_channel,out_channel):
        s = []
        if num ==0 :
            return
        for i in range(num):
            s.append(nn.Conv2d(1, out_channels=out_channel , kernel_size=(3, in_channel), stride=1, dilation=(i+1, 1), padding=(i+1, 0), bias=False).cuda())
            init.normal_(s[i].weight, std=0.001)
        return s
    def _get_TP_feature(self,x):
        s = []
        if self.temporal_conv==None:
            return
        x = x.permute(0,2,1).unsqueeze(1)
        for i in range(len(self.temporal_conv)):
            s.append(self.temporal_conv[i](x).squeeze(3).cuda())
        return s
    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        # pos =  PositionalEncoder_2(emb_dim = x.shape[1],max_len =x.shape[2],batch_size = batch_size )
        # x = x + pos().permute(0,2,1).cuda()
        # x = self.relu(self.tranpos_w(x))
        multi_att_list=[]
        temp_x = self._get_TP_feature(x)
        temp_x.append(x)
        for i in range(self.temporal_layer_num+1):
            temp_g_x = self.g[i](temp_x[i]).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            temp_theta_x = self.theta[i](temp_x[i]).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
            temp_phi_x = self.phi[i](temp_x[i]).view(batch_size, self.inter_channels, -1)
            temp_f = torch.matmul(temp_theta_x, temp_phi_x)
            temp_f = F.softmax(temp_f, dim=-1)
            temp_y = torch.matmul(temp_f, temp_g_x).permute(0, 2, 1).contiguous()
            temp_y = temp_y.view(batch_size, self.inter_channels, *x.size()[2:])
            multi_att_list.append(temp_y)
        y = torch.cat(multi_att_list,1)

        W_y = self.W(y)
        z = W_y + x
        z = self.f_x(z)
        return z

if __name__ == '__main__':
    import torch

    img = torch.zeros(2, 128, 20)
    net = NonLocalBlock1D_new_position_multi_head(in_channels=128, inter_channels=64,temporal_layer_num=2).cuda()
    out = net(torch.zeros(2, 128,10).cuda())
    print(out.size())