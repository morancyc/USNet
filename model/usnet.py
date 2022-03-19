import torch
from torch import nn
import torch.nn.functional as F
from model.backbone import symmetric_backbone
from model.module import ASPP
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


# Feature Compression and Adaptation block
class FCA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(x)
        x = torch.mul(input, x)
        return x


class Up_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.fca = FCA(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x, x_aux):
        x = F.interpolate(x, size=x_aux.size()[-2:], mode='bilinear')
        x_aux = self.fca(self.conv(x_aux))
        x = x + x_aux
        return x


# Multi-scale Evidence Collection Module
class MEC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.scale_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.scale_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        self.scale_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.softplus = nn.Softplus()

    def forward(self, x, input):
        e_1 = self.scale_1(x)
        e_2 = self.scale_2(x)
        e_3 = self.scale_3(x)

        e_1 = F.interpolate(e_1, size=input.size()[-2:], mode='bilinear')
        e_2 = F.interpolate(e_2, size=input.size()[-2:], mode='bilinear')
        e_3 = F.interpolate(e_3, size=input.size()[-2:], mode='bilinear')

        e_1 = self.softplus(e_1)
        e_2 = self.softplus(e_2)
        e_3 = self.softplus(e_3)

        e = (e_1 + e_2 + e_3) / 3
        return e_1, e_2, e_3, e


class USNet(nn.Module):
    def __init__(self, num_classes, backbone_name):
        super().__init__()
        # build backbone
        self.backbone = symmetric_backbone(name=backbone_name)

        if backbone_name == 'resnet101':
            # ASPP
            dilations = [1, 6, 12, 18]
            self.aspp_r = ASPP(2048, 256, dilations)
            self.aspp_d = ASPP(2048, 256, dilations)

            self.conv_r = ConvBlock(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
            self.conv_d = ConvBlock(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)

            self.up_r1 = Up_layer(in_channels=1024, out_channels=64)
            self.up_r2 = Up_layer(in_channels=512, out_channels=64)
            self.up_r3 = Up_layer(in_channels=256, out_channels=64)

            self.up_d1 = Up_layer(in_channels=1024, out_channels=64)
            self.up_d2 = Up_layer(in_channels=512, out_channels=64)
            self.up_d3 = Up_layer(in_channels=256, out_channels=64)

            self.mec_r = MEC(in_channels=64, out_channels=num_classes)
            self.mec_d = MEC(in_channels=64, out_channels=num_classes)

        elif backbone_name == 'resnet18':
            # ASPP
            dilations = [1, 6, 12, 18]
            self.aspp_r = ASPP(512, 256, dilations)
            self.aspp_d = ASPP(512, 256, dilations)

            self.conv_r = ConvBlock(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
            self.conv_d = ConvBlock(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)

            self.up_r1 = Up_layer(in_channels=256, out_channels=64)
            self.up_r2 = Up_layer(in_channels=128, out_channels=64)
            self.up_r3 = Up_layer(in_channels=64, out_channels=64)

            self.up_d1 = Up_layer(in_channels=256, out_channels=64)
            self.up_d2 = Up_layer(in_channels=128, out_channels=64)
            self.up_d3 = Up_layer(in_channels=64, out_channels=64)

            self.mec_r = MEC(in_channels=64, out_channels=num_classes)
            self.mec_d = MEC(in_channels=64, out_channels=num_classes)
        else:
            print('Error: unspported backbone \n')

        self.num_classes = num_classes
        self.softplus = nn.Softplus()
        self.init_weight()

    def init_weight(self):
        for name, m in self.named_modules():
            if 'backbone' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.num_classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.num_classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def forward(self, input_rgb, input_depth):
        # output of backbone
        x_r1, x_r2, x_r3, x_r4, x_d1, x_d2, x_d3, x_d4 = self.backbone(input_rgb, input_depth)

        # ASPP
        x_r4 = self.aspp_r(x_r4)
        x_d4 = self.aspp_d(x_d4)

        # compress channel to 64
        x_r4 = self.conv_r(x_r4)
        x_d4 = self.conv_d(x_d4)

        # decoder
        x_r3 = self.up_r1(x_r4, x_r3)
        x_r2 = self.up_r2(x_r3, x_r2)
        x_r1 = self.up_r3(x_r2, x_r1)

        x_d3 = self.up_d1(x_d4, x_d3)
        x_d2 = self.up_d2(x_d3, x_d2)
        x_d1 = self.up_d3(x_d2, x_d1)

        # MEC module
        e_r1, e_r2, e_r3, e_r = self.mec_r(x_r1, input_rgb)
        e_d1, e_d2, e_d3, e_d = self.mec_d(x_d1, input_depth)

        # compute evidence, alpha
        evidence = dict()
        evidence[0] = e_r.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*H*W, 2)
        evidence[1] = e_d.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*H*W, 2)

        alpha = dict()
        for v_num in range(len(evidence)):
            alpha[v_num] = evidence[v_num] + 1
        
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1

        if self.training == True:
            evidence_sup = dict()
            evidence_sup[0] = e_r1.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*H*W, 2)
            evidence_sup[1] = e_d1.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*H*W, 2)
            evidence_sup[2] = e_r2.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*H*W, 2)
            evidence_sup[3] = e_d2.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*H*W, 2)
            evidence_sup[4] = e_r3.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*H*W, 2)
            evidence_sup[5] = e_d3.permute(0, 2, 3, 1).reshape(-1, self.num_classes)  # (B*H*W, 2)

            alpha_sup = dict()
            for v_num in range(len(evidence_sup)):
                alpha_sup[v_num] = evidence_sup[v_num] + 1

            return evidence_sup, alpha_sup, evidence, evidence_a, alpha, alpha_a

        return evidence, evidence_a, alpha, alpha_a


if __name__ == '__main__':

    model = USNet(2, 'resnet18')
    model = model.cuda()
    summary(model, [(3, 1248, 384),(3, 1248, 384)])
