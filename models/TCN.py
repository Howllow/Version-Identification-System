import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """
    裁剪数据
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    使用一维卷积
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.8):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class TemperalBlock_without_res(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.8):
        super(TemperalBlock_without_res, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.net(x)
        return out
class TemperalBlock_without_res2(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.8):
        super(TemperalBlock_without_res2, self).__init__()
        self.conv1 =weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
        #                                    stride=stride, padding=padding, dilation=dilation)
        # self.chomp2 = Chomp1d(padding)
        # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.conv1.weight.data)
        torch.nn.init.constant_(self.conv1.weight.data, 0.0)

    def forward(self, x):
        out = self.net(x)
        return out
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
class TemporalConvNet_SPP(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet_SPP, self).__init__()
        layers = []

        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out_dic ={}
        #out_dic[0] = x
        for i in range(self.num_levels):
            if i == 0:
                out_dic[i] = self.network[i](x)
            else :
                out_dic[i] = self.network[i](out_dic[i-1])
        return out_dic
class TemporalNet_simple(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dilation_seed=1 ,dropout=0.2):
        super(TemporalNet_simple, self).__init__()
        layers = []

        self.num_levels = len(num_channels)
        for i in range(self.num_levels):
            dilation_size = dilation_seed
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemperalBlock_without_res2(in_channels, out_channels, kernel_size[i], stride=1, dilation=dilation_size,
                                     padding=0, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        self.maxpool1d = nn.MaxPool1d(2,stride=2)
    def forward(self, x):
        out_dic ={}
        #out_dic[0] = x
        for i in range(self.num_levels):
            if i == 0:
                out_dic[i] = self.maxpool1d(self.network[i](x))
            else :
                out_dic[i] = self.maxpool1d(self.network[i](out_dic[i-1]))
        for i in range(len(out_dic)):
            out_dic[i] = out_dic[i].permute(0, 2, 1)
        return out_dic
if __name__ =="__main__":
    s = TemporalConvNet_simple(num_inputs=512, num_channels=[512,512,512],
                            kernel_size=2,dilation_seed=1, dropout=0.5)
    out = s(torch.randn([1,512,37]))
    for i in range(len(out)):
        print(out[i].shape)