from scipy.special import binom
from models import *
from torch import Tensor
from typing import Tuple
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
           x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)


class SphereFace(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True, gamma=0.0):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.gamma = gamma
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input, target):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = torch.autograd.Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)

        return cos_theta, phi_theta


class SphereLoss(nn.Module):
    """"""
    def __init__(self, gamma=0):
        super(SphereLoss, self).__init__()
        self.gamma = gamma
        self.iter = 0
        self.lambda_min = 5.0
        self.lambda_max = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.iter += 1
        target = target.view(-1, 1)

        index = input[0].data * 0.0
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = Variable(index.byte())

        # Tricks
        # output(θyi) = (lambda * cos(θyi) + (-1) ** k * cos(m * θyi) - 2 * k)) / (1 + lambda)
        #             = cos(θyi) - cos(θyi) / (1 + lambda) + Phi(θyi) / (1 + lambda)
        self.lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.iter))
        output = input[0] * 1.0
        output[index] -= input[0][index] * 1.0 / (1 + self.lamb)
        output[index] += input[1][index] * 1.0 / (1 + self.lamb)

        # softmax loss
        logit = F.log_softmax(output)
        logit = logit.gather(1, target).view(-1)
        pt = logit.data.exp()

        loss = -1 * (1 - pt) ** self.gamma * logit
        loss = loss.mean()

        return loss


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):

        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 4.0 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for _, module in self.fc.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = F.normalize(module.weight, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        #print(list(self.fc.parameters()))
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin, device):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        self.device = device  # gpu or cpu

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1
        self.reset_parameters()

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        assert target is not None
        x, w = input, self.weight
        beta = max(self.beta, self.beta_min)
        logit = x.mm(w)
        indexes = range(logit.size(0))
        logit_target = logit[indexes, target]

        # cos(theta) = w * x / ||w||*||x||
        w_target_norm = w[:, target].norm(p=2, dim=0)
        x_norm = x.norm(p=2, dim=1)
        cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

        # equation 7
        cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

        # find k in equation 6
        k = self.find_k(cos_theta_target)

        # f_y_i
        logit_target_updated = (w_target_norm *
                                x_norm *
                                (((-1) ** k * cos_m_theta_target) - 2 * k))
        logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

        logit[indexes, target] = logit_target_updated_beta
        self.beta *= self.scale
        return logit


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    #print(normed_feature)
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    #print(similarity_matrix[positive_matrix], similarity_matrix[negative_matrix])
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class SMCNN_4_Angular(BasicModule):

    def __init__(self, loss_type, margin=3):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation_SPP_2()
        self.VGG_Conv1 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')
        self.fc = nn.Linear(18944, 2)
        self.adp_max_pool = torch.nn.AdaptiveMaxPool2d((1, None))
        self.fc1 = nn.Linear(512 + 256 + 128, 300)
        self.fc2 = nn.Linear(300, 10000)
        self.dropout = nn.Dropout(0.8)
        self.relu = torch.nn.ReLU()
        self.angular = False
        if loss_type == 'lsoftmax':
            self.angular_loss = LSoftmaxLinear(300, 10000, margin, 'cuda')
            self.angular = True
        elif loss_type == 'sphereface':
            self.angular_loss = SphereFace(300, 10000)
            self.angular = True
        elif loss_type != 'circlelabel' and loss_type != 'circlepair' and loss_type != 'base':
            self.angular_loss = AngularPenaltySMLoss(300, 10000, loss_type)
            self.angular = True
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
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4, seqb4).unsqueeze(1)


        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0], -1)
        VGG_out2 = self.VGG_Conv1(p_a3)
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0], -1)
        VGG_out3 = self.VGG_Conv1(p_a4)
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0], -1)
        VGG_out = torch.cat((VGG_out1, VGG_out2, VGG_out3), 1)
        samil = F.softmax(self.fc(VGG_out), dim=1)

        return samil[:, 1].unsqueeze(1)

    def get_fixed_out(self, seqa):
        seqa1, seqa2, seqa3, seqa4 = self.model(seqa)
        seqa2, seqa3, seqa4 = self.adp_max_pool(seqa2).squeeze(1), \
                              self.adp_max_pool(seqa3).squeeze(1), \
                              self.adp_max_pool(seqa4).squeeze(1)
        seqa = torch.cat((seqa2, seqa3, seqa4), 1)
        seqa = self.fc1(seqa)
        return seqa

    def forward(self, seqa, seqp, seqn, targeta, targetp, targetn):
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
        seqa1, seqp1, seqn1 = self.fc1(seqa), self.fc1(seqp), self.fc1(seqn)
        seqa_d, seqp_d, seqn_d = self.dropout(seqa1), self.dropout(seqp1), self.dropout(seqn1)


        return torch.cat((p_ap, p_an), dim=0), seqa1, seqp1, seqn1,\
                self.fc2(seqa_d), self.fc2(seqp_d), self.fc2(seqn_d), seqa_d, seqp_d, seqn_d


class MNISTNet(nn.Module):
    def __init__(self, margin, device):
        super(MNISTNet, self).__init__()
        self.margin = margin
        self.device = device

        self.conv_0 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, 3),
            nn.PReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(576, 256),
            nn.BatchNorm1d(256)
        )

        def forward(self, x, target=None):
            x = self.conv_0(x)
            x = self.conv_1(x)
            x = self.conv_2(x)
            x = self.conv_3(x)
            x = x.view(-1, 576)
            x = self.fc(x)
            return x
