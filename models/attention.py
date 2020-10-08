import numpy as np
import torch
from torch.nn import Module,  Conv2d, ReLU, Parameter, Softmax, Dropout


class Time_Module(Module):
    """ time attention module"""
    def __init__(self, in_dim, channel_num):
        super(Time_Module, self).__init__()
        self.channel_in = in_dim

        self.conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim * 8, kernel_size=(1, 1))

        self.conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim * 8, kernel_size=(1, 1))

        self.gamma = Parameter(torch.zeros(1))


        self.softmax = Softmax(dim=2)

    def forward(self, x):
        """
        :param x: input feature maps( B X C X H X W)
        :return: out : attention value + input feature
        """
        mini_batchsize, C, Height, Width = x.size()
        proj_query = self.conv1(x).permute(0,2,1,3).reshape(mini_batchsize, Height, 8 * Width)
        proj_key = self.conv2(x).permute(0,1,3,2).reshape(mini_batchsize, 8 * Width, Height)
        energy = torch.matmul(proj_query, proj_key)

        max_H, _ = energy.max(dim=2, keepdim=True)
        min_H, _ = energy.min(dim=2, keepdim=True)
        temp_b = (energy - min_H)
        temp_c = (max_H - min_H)+0.00000001
        energy = temp_b / temp_c
        attention = self.softmax(energy)

        attention = attention.reshape(mini_batchsize, 1, Height, Height)

        out = torch.matmul(attention, x)

        out = self.gamma*out + x
        return out

class Channel_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(Channel_Module, self).__init__()
        self.chanel_in = in_dim

        self.conv1 = Conv2d(in_channels=in_dim, out_channels=in_dim * 8, kernel_size=(1, 1))

        self.conv2 = Conv2d(in_channels=in_dim, out_channels=in_dim * 8, kernel_size=(1, 1))

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=2)


    def forward(self, x):
        """
        :param x: input feature maps( B X C X H X W)
        :return: out : attention value + input feature
        """
        mini_batchsize, C, Height, Width = x.size()
        queryv = self.conv1(x).permute(0,3,1,2).reshape(mini_batchsize, Width, 8 * Height)
        key = self.conv2(x).reshape(mini_batchsize, 8 * Height, Width)
        energy = torch.matmul(queryv, key)

        max_H, _ = energy.max(dim=2, keepdim=True)
        min_H, _ = energy.min(dim=2, keepdim=True)
        temp_b = (energy - min_H)
        temp_c = (max_H - min_H)+0.00000001
        energy = temp_b / temp_c
        attention = self.softmax(energy)

        attention = attention.reshape(mini_batchsize, 1, Width, Width)
        out = torch.matmul(x, attention.permute(0,1,3,2))

        out = self.gamma * out + x

        return out

class Attention(Module):
    def __init__(self, in_dim, channel_num):
        super(Attention, self).__init__()
        self.ca = Channel_Module(in_dim)
        self.ta = Time_Module(in_dim, channel_num)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 1, stride=1),
            torch.nn.BatchNorm2d(16, momentum=0.1, affine=True),
        )

    def forward(self, x):
        out1 = self.ca(x)
        out2 = self.ta(x)
        out = torch.cat((x, out1, out2), dim=1)


        return out





