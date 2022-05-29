



import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from collections import OrderedDict

from tensorboardX import SummaryWriter

from torchvision.utils import make_grid
from ..attention.CBAM import CBAMBlock
from ..attention.BAM import BAMBlock
from ..attention.DANet import DAModule
from ..attention.ECAAttention import ECAAttention
from ..attention.ShuffleAttention import ShuffleAttention
from ..attention.A2Atttention import DoubleAttention
from ..attention.SGE import SpatialGroupEnhance

"""
CBAM
CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
"""


class SKAttention(nn.Module):

    def __init__(self, channel=256, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


class SEAttention(nn.Module):

    def __init__(self, channel=256, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)





class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super(ExternalAttention, self).__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        #         print(attn.shape)
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model

        return out


class EAttention(nn.Module):
    def __init__(self, si):
        super(EAttention, self).__init__()

        self.conv3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.ea = ExternalAttention(si)

    def forward(self, N, P):
        #         print("shiyongjiN",N.shape)
        x = self.conv3x3(N)
        #         print("shiyongjix",x.shape)
        p_n_add = x + P
        out = self.ea(p_n_add)
        return out


class ExternaAtt(nn.Module):
    def __init__(self):
        super(ExternaAtt, self).__init__()

        self.et1 = EAttention(96)
        self.et2 = EAttention(48)
        self.et3 = EAttention(24)

    def forward(self, p_list):
        P5 = p_list[3]  # 24
        P4 = p_list[2]  # 48
        P3 = p_list[1]  # 96
        P2 = p_list[0]  # 192

        N2 = P2
        N3 = self.et1(N2, P3)
        N4 = self.et2(N3, P4)
        N5 = self.et3(N4, P5)

        return [N2, N3, N4, N5]




class down_to_up_only(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(down_to_up_only, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, N, P):
        x = self.conv3x3(N)
        out = x + P
        return out


class down_to_up_only_net(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(down_to_up_only_net, self).__init__()
        self.downtoup1 = down_to_up_only(in_planes, out_planes)
        self.downtoup2 = down_to_up_only(in_planes, out_planes)
        self.downtoup3 = down_to_up_only(in_planes, out_planes)

    def forward(self, p_list):
        P5 = p_list[3]
        P4 = p_list[2]
        P3 = p_list[1]
        P2 = p_list[0]
        N2 = P2
        N3 = self.downtoup1(N2, P3)
        N4 = self.downtoup2(N3, P4)
        N5 = self.downtoup3(N4, P5)

        return [N2, N3, N4, N5]




class down_to_up_skip(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(down_to_up_skip, self).__init__()
        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=1, bias=False)  #

    def forward(self, N, P):
        x = self.conv3x3(N)
        out = x + P
        return out


class down_to_up_skip_net(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(down_to_up_skip_net, self).__init__()
        self.downtoup1 = down_to_up_only(in_planes, out_planes)
        self.downtoup2 = down_to_up_only(in_planes, out_planes)
        self.downtoup3 = down_to_up_only(in_planes, out_planes)

    def forward(self, p_list, q_list):
        P5 = p_list[3]
        P4 = p_list[2]
        P3 = p_list[1]
        P2 = p_list[0]

        Q5 = q_list[3]
        Q4 = q_list[2]
        Q3 = q_list[1]
        Q2 = q_list[0]

        N2 = P2 + Q2
        N3 = self.downtoup1(N2, P3 + Q3)
        N4 = self.downtoup2(N3, P4 + Q4)
        N5 = self.downtoup3(N4, P5 + Q5)

        return [N2, N3, N4, N5]


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



class down_to_up_skip_1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(down_to_up_skip_1, self).__init__()
        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, N, P):
        x = self.conv3x3(N)
        out = x + P
        return out


class down_to_up_skip_1_net(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(down_to_up_skip_1_net, self).__init__()
        self.downtoup1 = down_to_up_skip_1(in_planes, out_planes)
        self.downtoup2 = down_to_up_skip_1(in_planes, out_planes)
        self.downtoup3 = down_to_up_skip_1(in_planes, out_planes)
        self.conv1x1_1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.conv1x1_2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.conv1x1_3 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.conv1x1_4 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

    def forward(self, p_list, q_list):
        P5 = p_list[3]
        P4 = p_list[2]
        P3 = p_list[1]
        P2 = p_list[0]

        #         Q5 = self.conv1x1_1(q_list[3])
        #         Q4 = self.conv1x1_2(q_list[2])
        #         Q3 = self.conv1x1_3(q_list[1])
        #         Q2 = self.conv1x1_4(q_list[0])

        Q5 = q_list[3]
        Q4 = q_list[2]
        Q3 = q_list[1]
        Q2 = q_list[0]

        N2 = P2 + Q2
        N3 = self.downtoup1(N2, P3 + Q3)
        N4 = self.downtoup2(N3, P4 + Q4)
        N5 = self.downtoup3(N4, P5 + Q5)

        return [N2, N3, N4, N5]


class ChAttModule(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ChAttModule, self).__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.conv1x1(x)
        x = self.softmax(x)  # 使用 softmax

        return x



class ChAttModule_1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ChAttModule_1, self).__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.swish = Swish()
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.se = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, 1, bias=False),
            Swish(),
            nn.Conv2d(in_planes, in_planes, 1, bias=False)
        )

    def forward(self, x):
        a = self.global_pool(x)
        b = self.avgpool(x)
        a = self.se(a)
        b = self.se(b)
        #         x = self.conv1x1(x)
        x = self.softmax(a + b)  # 使用 softmax

        return x

class Botblock_CA(nn.Module):
    def __init__(self, in_planes, out_planes, ):
        super(Botblock_CA, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)


        self.channel_attention = ChAttModule(in_planes, out_planes)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add
        ca_feature = self.channel_attention(out)
        out = ca_feature * out + out

        return out


class SAttModule(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SAttModule, self).__init__()
        self.conv_to_1 = nn.Sequential(
            nn.Conv2d(in_planes, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_to_1(x)

        x = self.softmax(x)
        return x


class SAttModule_1(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SAttModule, self).__init__()
        self.conv_to_1 = nn.Sequential(
            nn.Conv2d(in_planes, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_to_1(x)

        x = self.softmax(x)
        return x


class Botblock_SA(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_SA, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)

        self.s_attention = SAttModule(in_planes, 1)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add
        sa_feature = self.s_attention(out)
        out = sa_feature * out + out

        return out


class Botblock_SA2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_SA2, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)

        self.s_attention = SAttModule(in_planes, 1)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add
        sa_feature = self.s_attention(out)
        out = sa_feature * out

        return out




class Botblock_SA_CA(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_SA_CA, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        self.channel_attention = ChAttModule(in_planes, out_planes)

        self.s_attention = SAttModule(in_planes, 1)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add


        sa_feature = self.s_attention(out)
        sa_out = sa_feature * out + out

        ca_feature = self.channel_attention(sa_out)
        out = ca_feature * sa_out + sa_out
        return out


class Botblock_CA_SA(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_CA_SA, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        self.channel_attention = ChAttModule(in_planes, out_planes)

        self.s_attention = SAttModule(in_planes, 1)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add


        ca_feature = self.channel_attention(out)
        ca_out = ca_feature * out + out

        sa_feature = self.s_attention(ca_out)
        out = sa_feature * ca_out + ca_out

        return out


class Botblock_CA_SA2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_CA_SA2, self).__init__()


        self.channel_attention = ChAttModule(in_planes, out_planes)

        self.s_attention = SAttModule(in_planes, 1)


    def forward(self, x):
        # x = self.conv3x3(N)
        # p_n_add = x + P
        out = x

        ca_feature = self.channel_attention(out)
        out = ca_feature * out + out

        sa_feature = self.s_attention(out)
        out = sa_feature * out 
        return out


class BotUpNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BotUpNet, self).__init__()
        self.ca_sa_block1 = Botblock_CA_SA2(in_planes, out_planes)
        self.ca_sa_block2 = Botblock_CA_SA2(in_planes, out_planes)
        self.ca_sa_block3 = Botblock_CA_SA2(in_planes, out_planes)

    def forward(self, p_list):
        P5 = p_list[3]  # 24
        P4 = p_list[2]  # 48
        P3 = p_list[1]  # 96
        P2 = p_list[0]  # 192

        N2 = P2
        N3 = self.ca_sa_block1(N2, P3)
        N4 = self.ca_sa_block2(N3, P4)
        N5 = self.ca_sa_block3(N4, P5)

        return [N2, N3, N4, N5]



======================================================