import torch
import torch.nn as nn

from NeuralBlocks import BaseBlock, SimpleDecoder, SimpleEncoder, UpsampleDeConv, ResidualBlock


'''
Image Restoration Using Convolutional
Auto-encoders with Symmetric Skip
Connections
Xiao-Jiao Mao, Chunhua Shen, Yu-Bin Yang
'''

class AdelaideGenerator(torch.nn.Module):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(AdelaideGenerator, self).__init__()
        self.enc1 = SimpleEncoder(dimension, 64, activation)
        self.enc2 = SimpleEncoder(64,  128, activation)
        self.enc3 = SimpleEncoder(128, 256, activation)
        self.enc4 = SimpleEncoder(256, 512, activation)
        self.enc5 = SimpleEncoder(512, 512, activation)
        self.dec1 = SimpleDecoder(512, 512, deconv=deconv)
        self.dec2 = SimpleDecoder(512, 256, deconv=deconv)
        self.dec3 = SimpleDecoder(256, 128, deconv=deconv)
        self.dec4 = SimpleDecoder(128, 64,  deconv=deconv)
        self.dec5 = SimpleDecoder(64, dimension, deconv=deconv)
        self.activation = activation
        self.deconv1 = deconv

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        c = self.activation(e5)
        d1 = self.dec1(c)
        c = self.activation(torch.add(d1, e4))
        d2 = self.dec2(c)
        c = self.activation(torch.add(d2, e3))
        d3 = self.dec3(c)
        c = self.activation(torch.add(d3, e2))
        d4 = self.dec4(c)
        c = self.activation(torch.add(d4, e1))
        y  = self.dec5(c)
        return torch.tanh(y)


class AdelaideFastGenerator(AdelaideGenerator):
    def __init__(self,dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(AdelaideFastGenerator, self).__init__(dimension, deconv, activation, drop_out)
        self.enc1 = ResidualBlock(dimension, 16, stride = 2, activation=activation)
        self.enc2 = ResidualBlock(16,  32,      stride = 2,  activation=activation)
        self.enc3 = ResidualBlock(32, 64,       stride = 2,   activation=activation)
        self.enc4 = ResidualBlock(64, 128,      stride = 2,  activation=activation)
        self.enc5 = ResidualBlock(128, 256,     stride = 2, activation=activation)
        self.dec1 = SimpleDecoder(256, 128,         deconv=deconv)
        self.dec2 = SimpleDecoder(128, 64,          deconv=deconv)
        self.dec3 = SimpleDecoder(64, 32,           deconv=deconv)
        self.dec4 = SimpleDecoder(32, 16,           deconv=deconv)
        self.dec5 = SimpleDecoder(16, dimension,    deconv=deconv)


class AdelaideResidualGenerator(AdelaideGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(AdelaideResidualGenerator, self).__init__(dimension, deconv, activation, drop_out)
        self.enc1 = ResidualBlock(dimension, 64, stride = 2,activation=activation)
        self.enc2 = ResidualBlock(64,  128,     stride = 2,activation=activation)
        self.enc3 = ResidualBlock(128, 256,     stride = 2,activation=activation)
        self.enc4 = ResidualBlock(256, 512,     stride = 2,activation=activation)
        self.enc5 = ResidualBlock(512, 512,     stride = 2,activation=activation)


class AdelaideSupremeGenerator(AdelaideResidualGenerator):
    def __init__(self,dimension,  deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(AdelaideSupremeGenerator, self).__init__(dimension, deconv, activation, drop_out)
        self.skip1 = BaseBlock(64, 64, 3, 1, activation=activation)
        self.skip2 = BaseBlock(128, 128, 3, 1, activation=activation)
        self.skip3 = BaseBlock(256, 256, 3, 1, activation=activation)
        self.skip4 = BaseBlock(512, 512, 3, 1, activation=activation)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d1 = self.dec1(e5)
        c = self.activation(torch.add(d1, self.skip4(e4)))
        d2 = self.dec2(c)
        c = self.activation(torch.add(d2, self.skip3(e3)))
        d3 = self.dec3(c)
        c = self.activation(torch.add(d3, self.skip2(e2)))
        d4 = self.dec4(c)
        c = self.activation(torch.add(d4, self.skip1(e1)))
        y = self.dec5(c)
        return torch.tanh(y)
