from scipy.special import binom
from models import *

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
            self.m = 1.35 if not m else m
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

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

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
        self.loss_func = torch.nn.CrossEntropyLoss()

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
        return self.loss_func(logit, target)


class SMCNN_4_Angular(BasicModule):

    def __init__(self, loss_type, margin=2):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation_SPP_2()
        self.VGG_Conv1 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')
        self.fc = nn.Linear(18944, 2)
        self.adp_max_pool = torch.nn.AdaptiveMaxPool2d((1, None))
        self.fc1 = nn.Sequential(
                    nn.Linear(512 + 256 + 128, 300),
                    nn.BatchNorm1d(300))
        self.dropout = nn.Dropout(0.7)
        self.relu = torch.nn.ReLU()
        if loss_type == 'lsoftmax':
            self.angular_loss = LSoftmaxLinear(300, 80, margin, 'cuda')
        else:
            self.angular_loss = AngularPenaltySMLoss(300, 80, loss_type)

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
        seqa, seqp, seqn = self.fc1(seqa), self.fc1(seqp), self.fc1(seqn)
        seqa_d, seqp_d, seqn_d = self.dropout(seqa), self.dropout(seqp), self.dropout(seqn)
        loss = self.angular_loss(seqa, targeta) + \
               self.angular_loss(seqp, targetp) + \
               self.angular_loss(seqn, targetn)

        return torch.cat((p_ap, p_an), dim=0), seqa, seqp, seqn, loss



