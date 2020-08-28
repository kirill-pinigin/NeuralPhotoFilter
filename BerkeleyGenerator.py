import torch
import torch.nn as nn

from NeuralBlocks import BaseBlock, SimpleDecoder, SimpleEncoder, UpsampleDeConv, ResidualBlock

LATENT_SPACE = int(64)

'''
Image-to-Image Translation with Conditional Adversarial Networks
Phillip Isola Jun-Yan Zhu Tinghui Zhou Alexei A. Efros
Berkeley AI Research (BAIR) Laboratory, UC Berkeley
'''

class BerkeleyGenerator(torch.nn.Module):
    def __init__(self,dimension,  deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(BerkeleyGenerator, self).__init__()
        self.layer1  = SimpleEncoder(dimension, 64, bn = False)
        self.layer2  = SimpleEncoder(64,  128)
        self.layer3  = SimpleEncoder(128, 256)
        self.layer4  = SimpleEncoder(256, 512)
        self.layer5  = SimpleEncoder(512, 512)
        self.layer6  = SimpleEncoder(512, 512)

        self.layer7  = SimpleDecoder(512,  512, deconv=deconv)
        self.layer8  = SimpleDecoder(1024, 256, deconv=deconv)
        self.layer9  = SimpleDecoder(768,  128, deconv=deconv)
        self.layer10 = SimpleDecoder(384,   64, deconv=deconv)
        self.layer11 = SimpleDecoder(192,   64, deconv=deconv)
        self.layer12 = SimpleDecoder(64, dimension, deconv=deconv)

        self.activation = activation
        self.deconv1 = deconv

    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)

        l7 = self.layer7(l6)
        c = self.activation(torch.cat([l7, l5], dim=1))
        l8 = self.layer8(c)
        c = self.activation(torch.cat([l8, l4], dim=1))
        l9 = self.layer9(c)
        c = self.activation(torch.cat([l9, l3], dim=1))
        l10 = self.layer10(c)
        c = self.activation(torch.cat([l10, l2], dim=1))
        l11 = self.layer11(c)
        y = self.activation(l11)
        y  = self.layer12(y)
        return torch.tanh(y)


class BerkeleyFastGenerator(BerkeleyGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(BerkeleyFastGenerator, self).__init__(dimension, deconv, activation, drop_out)
        self.layer1  = ResidualBlock(dimension, 16,  stride = 2, activation=activation)
        self.layer2  = ResidualBlock(16,  32,       stride = 2, activation=activation)
        self.layer3  = ResidualBlock(32, 64,        stride = 2, activation=activation)
        self.layer4  = ResidualBlock(64, 128,       stride = 2, activation=activation)
        self.layer5  = ResidualBlock(128, 256,      stride = 2, activation=activation)
        self.layer6  = ResidualBlock(256, 512,      stride = 2, activation=activation)
        self.layer7  = SimpleDecoder(512 * 1, 256, deconv=deconv)
        self.layer8  = SimpleDecoder(256 * 2, 128, deconv=deconv)
        self.layer9  = SimpleDecoder(128 * 2, 64, deconv=deconv)
        self.layer10 = SimpleDecoder(64 * 2, 32, deconv=deconv)
        self.layer11 = SimpleDecoder(32 * 2, 16, deconv=deconv)
        self.layer12 = SimpleDecoder(16 , dimension, deconv=deconv)


class BerkeleyResidualGenerator(BerkeleyGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(BerkeleyResidualGenerator, self).__init__(dimension, deconv, activation, drop_out)
        self.layer1  = ResidualBlock(dimension, 64,  stride = 2, activation = activation)
        self.layer2  = ResidualBlock(64,  128,      stride = 2, activation = activation)
        self.layer3  = ResidualBlock(128, 256,      stride = 2, activation = activation)
        self.layer4  = ResidualBlock(256, 512,      stride = 2, activation = activation)
        self.layer5  = ResidualBlock(512, 512,      stride = 2, activation = activation)
        self.layer6  = ResidualBlock(512, 512,      stride = 2, activation = activation)


class BerkeleySupremeGenerator(BerkeleyResidualGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(BerkeleySupremeGenerator, self).__init__(dimension, deconv, activation, drop_out)
        self.skip1 = BaseBlock(128, 128, 3, 1, activation=activation)
        self.skip2 = BaseBlock(256, 256, 3, 1, activation=activation)
        self.skip3 = BaseBlock(512, 512, 3, 1, activation=activation)
        self.skip4 = BaseBlock(512, 512, 3, 1, activation=activation)

    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)

        l7 = self.layer7(l6)
        c = self.activation(torch.cat([l7, self.skip4(l5)], dim=1))
        l8 = self.layer8(c)
        c = self.activation(torch.cat([l8, self.skip3(l4)], dim=1))
        l9 = self.layer9(c)
        c = self.activation(torch.cat([l9, self.skip2(l3)], dim=1))
        l10 = self.layer10(c)
        c = self.activation(torch.cat([l10, self.skip1(l2)], dim=1))
        l11 = self.layer11(c)
        y = self.activation(l11)
        y = self.layer12(y)
        return torch.tanh(y)
