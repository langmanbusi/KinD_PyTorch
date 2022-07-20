import os
import time
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from utils import MSE, SSIM, PSNR, LPIPS
from torch.utils.tensorboard import SummaryWriter
import atexit
import logging


class Restoration_net(nn.Module):
    def __init__(self, inchannel=4, kernel_size=(3, 3), padding=(1, 1)):
        super(Restoration_net, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2)
        self.restore_conv1 = nn.Sequential(
            nn.Conv2d(inchannel, 32, kernel_size, padding=padding, padding_mode='replicate'), self.lrelu,
            nn.Conv2d(32, 32, kernel_size, padding=padding, padding_mode='replicate'), self.lrelu
        )

        self.restore_conv2 = nn.Sequential(
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(32, 64, kernel_size, padding=padding), self.lrelu,
            nn.Conv2d(64, 64, kernel_size, padding=padding), self.lrelu
        )

        self.restore_conv3 = nn.Sequential(
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, kernel_size, padding=padding), self.lrelu,
            nn.Conv2d(128, 128, kernel_size, padding=padding), self.lrelu
        )

        self.restore_conv4 = nn.Sequential(
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(128, 256, kernel_size, padding=padding), self.lrelu,
            nn.Conv2d(256, 256, kernel_size, padding=padding), self.lrelu
        )

        self.restore_conv5 = nn.Sequential(
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(256, 512, kernel_size, padding=padding), self.lrelu,
            nn.Conv2d(512, 512, kernel_size, padding=padding), self.lrelu
        )

        self.restore_upsample1 = nn.ConvTranspose2d(512, 256, (2, 2), (2, 2))
        self.restore_upsample2 = nn.ConvTranspose2d(256, 128, (2, 2), (2, 2))
        self.restore_upsample3 = nn.ConvTranspose2d(128, 64, (2, 2), (2, 2))
        self.restore_upsample4 = nn.ConvTranspose2d(64, 32, (2, 2), (2, 2))

        self.restore_deconv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size, padding=padding), self.lrelu,
            nn.Conv2d(256, 256, kernel_size, padding=padding), self.lrelu
        )

        self.restore_deconv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size, padding=padding), self.lrelu,
            nn.Conv2d(128, 128, kernel_size, padding=padding), self.lrelu
        )

        self.restore_deconv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size, padding=padding), self.lrelu,
            nn.Conv2d(64, 64, kernel_size, padding=padding), self.lrelu
        )

        self.restore_deconv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size, padding=padding), self.lrelu,
            nn.Conv2d(32, 32, kernel_size, padding=padding), self.lrelu,
            nn.Conv2d(32, 3, kernel_size, padding=padding), nn.Sigmoid()
        )

    def forward(self, input_R, input_L):
        input_all = torch.cat((input_R, input_L), dim=1)
        conv1 = self.restore_conv1(input_all)
        conv2 = self.restore_conv2(conv1)   # downsample * 2
        conv3 = self.restore_conv3(conv2)
        conv4 = self.restore_conv4(conv3)
        conv5 = self.restore_conv5(conv4)

        up1 = torch.cat((self.restore_upsample1(conv5), conv4), dim=1)
        deconv1 = self.restore_deconv1(up1)
        up2 = torch.cat((self.restore_upsample2(deconv1), conv3), dim=1)
        deconv2 = self.restore_deconv2(up2)
        up3 = torch.cat((self.restore_upsample3(deconv2), conv2), dim=1)
        deconv3 = self.restore_deconv3(up3)
        up4 = torch.cat((self.restore_upsample4(deconv3), conv1), dim=1)
        deconv4 = self.restore_deconv4(up4)

        return deconv4


class DecomNet(nn.Module):
    def __init__(self, channel=32, kernel_size=(3, 3), padding=(1, 1)):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.lrelu = nn.LeakyReLU(0.2)
        # Decom R
        self.decom_conv1 = nn.Sequential(
            nn.Conv2d(3, channel, kernel_size, padding=padding, padding_mode='replicate'), self.lrelu
        )
        self.decom_conv2 = nn.Sequential(
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(channel, channel*2, kernel_size, padding=padding), self.lrelu
        )
        self.decom_conv3 = nn.Sequential(
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(channel*2, channel*4, kernel_size, padding=padding), self.lrelu
        )
        self.decom_upsample1 = nn.ConvTranspose2d(channel*4, channel*2, (2, 2), (2, 2))
        self.decom_conv4 = nn.Sequential(
            nn.Conv2d(channel*4, channel*2, kernel_size, padding=padding, padding_mode='replicate'), self.lrelu
        )
        self.decom_upsample2 = nn.ConvTranspose2d(channel*2, channel, (2, 2), (2, 2))
        self.decom_conv5 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size, padding=padding, padding_mode='replicate'), self.lrelu
        )
        self.decom_out_R = nn.Sequential(
            nn.Conv2d(channel, 3, (1, 1)), nn.Sigmoid()
        )
        # Decom I
        self.decom_conv6 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=padding), self.lrelu
        )
        self.decom_out_I = nn.Sequential(
            nn.Conv2d(channel * 2, 1, (1, 1)), nn.Sigmoid()
        )

    def forward(self, input_im):
        conv1 = self.decom_conv1(input_im)
        conv2 = self.decom_conv2(conv1)
        conv3 = self.decom_conv3(conv2)
        up1 = torch.cat((self.decom_upsample1(conv3), conv2), dim=1)
        conv4 = self.decom_conv4(up1)
        up2 = torch.cat((self.decom_upsample2(conv4), conv1), dim=1)
        conv5 = self.decom_conv5(up2)
        out_R = self.decom_out_R(conv5)

        conv6 = self.decom_conv6(conv1)
        out_I = self.decom_out_I(torch.cat((conv6, conv5), dim=1))

        return out_R, out_I


# KinD RelightNet
class RelightNet(nn.Module):
    def __init__(self, channel=32, kernel_size=(3, 3), padding=(1, 1)):
        super(RelightNet, self).__init__()

        self.lrelu = nn.LeakyReLU(0.2)
        self.relight_conv6 = nn.Sequential(
            nn.Conv2d(2, channel, kernel_size, padding=padding, padding_mode='replicate'), self.lrelu,
            nn.Conv2d(channel, channel, kernel_size, padding=padding, padding_mode='replicate'), self.lrelu
        )
        self.relight_conv7 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=padding, padding_mode='replicate'), self.lrelu,
            nn.Conv2d(channel, channel, kernel_size, padding=padding, padding_mode='replicate'), self.lrelu,
            nn.Conv2d(channel, 1, (1, 1))
        )
        self.out = nn.Sigmoid()

    def forward(self, input_I, input_R, ratio):
        input_L = torch.cat((input_I, ratio), dim=1)

        conv6 = self.relight_conv6(input_L)
        conv7 = self.relight_conv7(conv6)
        out = self.out(conv7)

        return out


class DenoiseNet_No_Seg(nn.Module):
    def __init__(self, is_training=True, layer_num=15, channel=64):
        super(DenoiseNet_No_Seg, self).__init__()

        self.rdn = RDN_No_Seg(3, 64)

    def forward(self, input_R, input_I):
        out_R = self.rdn(input_R, input_I)

        return out_R


class RDN_No_Seg(nn.Module):
    def __init__(self, nChannel, nfeat, nResBlock=4, growthRate=0):
        super(RDN_No_Seg, self).__init__()
        self.conv1 = nn.Conv2d(4, nfeat, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(nfeat, nfeat, kernel_size=3, padding=1)

        self.RDB1 = RDB(nfeat, nfeat, nResBlock)
        self.RIRB1 = RIRB_No_Seg(nfeat, nfeat)
        self.block1 = nn.Sequential(
            RDB(nfeat, nfeat, nResBlock), RDB(nfeat, nfeat, nResBlock)
        )
        self.RDB2 = RDB(2 * nfeat, nfeat, nResBlock)
        self.RIRB2 = RIRB_No_Seg(nfeat, nfeat)
        self.block2 = nn.Sequential(
            RDB(nfeat, nfeat, nResBlock), RDB(nfeat, nfeat, nResBlock)
        )

        self.conv3 = nn.Conv2d(3 * nfeat, nfeat, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(nfeat, nfeat, kernel_size=3, padding=1)
        self.block3 = nn.Sequential(
            nn.Conv2d(nfeat, nfeat, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(nfeat, nChannel, kernel_size=3, padding=1)
        )

    def forward(self, input_R, input_I):
        input_all = torch.cat([input_R, input_I], dim=1)
        F_ = self.conv1(input_all)
        F_0 = self.RIRB1(self.RDB1(self.conv2(F_)))
        # F_0 = self.RDB1(self.conv2(F_))
        F_4 = self.block1(F_0)
        F_F1 = torch.cat([F_4, F_], dim=1)
        F_F1 = self.RIRB2(self.RDB2(F_F1))
        # F_F1 = self.RDB2(F_F1)
        F_8 = self.block2(F_F1)
        FF = torch.cat([F_0, F_F1, F_8], dim=1)
        FGF = self.conv4(self.conv3(FF))
        out = self.block3(FGF)

        return out


class RIRB_No_Seg(nn.Module):
    def __init__(self, inchannel, nChannels):
        super(RIRB_No_Seg, self).__init__()

        self.conv1 = nn.Conv2d(inchannel, nChannels, kernel_size=1, padding=0)
        self.resblocks1 = nn.ModuleList()
        for i in range(2):
            self.resblocks1.append(ResidualBlock(nChannels, nChannels))

        self.conv2 = nn.Conv2d(nChannels, nChannels, kernel_size=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(nChannels, nChannels, kernel_size=1, padding=0)

        self.resblocks2 = nn.ModuleList()
        for i in range(2):
            self.resblocks2.append(ResidualBlock(nChannels, nChannels))

        self.conv4 = nn.Conv2d(inchannel + nChannels, nChannels, kernel_size=1, padding=0)

    def forward(self, input_R):
        out = self.conv1(input_R)
        for resblock1 in self.resblocks1:
            out = resblock1(out)

        out = self.conv2(out)
        out = self.conv3(out)
        for resblock2 in self.resblocks2:
            out = resblock2(out)

        out = torch.cat([out, input_R], dim=1)
        out = self.conv4(out)

        return out


class RDB(nn.Module):
    def __init__(self, inchannel, nChannels, nResBlock):
        super(RDB, self).__init__()

        self.conv1 = nn.Conv2d(inchannel, nChannels, kernel_size=1, padding=0)
        self.resblocks = nn.ModuleList()
        for i in range(nResBlock):
            self.resblocks.append(ResidualBlock(nChannels, nChannels))

        self.conv2 = nn.Conv2d(inchannel + nChannels, nChannels, kernel_size=1, padding=0)

    def forward(self, input_R):
        out = self.conv1(input_R)
        for resblock in self.resblocks:
            out = resblock(out)

        out = torch.cat([out, input_R], dim=1)
        out = self.conv2(out)

        return out


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None,
                 BN_momentum=0.1):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Mymodel(nn.Module):
    def __init__(self, gpu):
        super(Mymodel, self).__init__()

        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()
        self.RestoreNet = Restoration_net()
        # self.DenoiseNet = DenoiseNet_No_Seg()
        if gpu is not None:
            gpu = 0
            self.gpu = torch.device('cuda:' + str(gpu))

    def forward(self, input_low, input_high, ratio):
        # Forward DecompNet
        R_low, I_low = self.DecomNet(input_low)
        R_high, I_high = self.DecomNet(input_high)

        # Forward RestoreNet
        R_restore = self.RestoreNet(R_low, I_low)
        # Forward DenoiseNet
        # R_denoise = self.DenoiseNet(R_low, I_low)

        # Forward RelightNet
        I_delta = self.RelightNet(I_low, R_low, ratio)

        # Other variables
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)
        R_denoise = R_restore

        return input_low, input_high, R_low, I_low, R_high, I_high, I_delta, I_low_3, I_high_3, I_delta_3, R_denoise

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters: {}'.format(params))
        print(self)
