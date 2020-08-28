import torch
import torch.nn as nn

from NeuralBlocks import ResidualBlock, ConvLayer, UpsampleDeConv, BaseBlock

LATENT_SPACE   = int(128)
LATENT_SPACE_2 = int(LATENT_SPACE / 2)
LATENT_SPACE_4 = int(LATENT_SPACE / 4)
LATENT_SPACE_8 = int(LATENT_SPACE / 8)

'''
Perceptual Losses for Real-Time Style Transfer
and Super-Resolution
Justin Johnson, Alexandre Alahi, Li Fei-Fei
{jcjohns, alahi, feifeili}@cs.stanford.edu
Department of Computer Science, Stanford University
'''

class StanfordGenerator(torch.nn.Module):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(StanfordGenerator, self).__init__()
        self.DEPTH_SIZE = int(5)

        self.conv1 = BaseBlock(dimension, LATENT_SPACE_4, 9, 1, activation=activation, drop_out=drop_out)
        self.conv2 = BaseBlock(LATENT_SPACE_4, LATENT_SPACE_2, 3, 2, activation=activation, drop_out=drop_out)
        self.conv3 = BaseBlock(LATENT_SPACE_2, LATENT_SPACE, 3, 2, activation=activation, drop_out=drop_out)
        self.residual_blocks = nn.Sequential()

        for i in range(0, self.DEPTH_SIZE):
            self.residual_blocks.add_module(str(i), ResidualBlock(LATENT_SPACE, LATENT_SPACE, stride=1, activation=activation, drop_out=drop_out))

        self.deconv1 = deconv(LATENT_SPACE, LATENT_SPACE_2)
        self.norm1 = torch.nn.BatchNorm2d(LATENT_SPACE_2, affine=True)
        self.deconv2 = deconv(LATENT_SPACE_2, LATENT_SPACE_4)
        self.norm2 = torch.nn.BatchNorm2d(LATENT_SPACE_4, affine=True)
        self.final = ConvLayer(LATENT_SPACE_4, dimension, kernel_size=9, stride=1)
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.residual_blocks(x)
        x = self.activation(self.norm1(self.deconv1(x)))
        x = self.activation(self.norm2(self.deconv2(x)))
        x = self.final(x)
        return torch.tanh(x)


class StanfordFastGenerator(torch.nn.Module):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(StanfordFastGenerator, self).__init__()
        self.DEPTH_SIZE = int(3)
        self.conv1 = BaseBlock(dimension, LATENT_SPACE_8, 9, 1, activation=activation, drop_out=drop_out)
        self.conv2 = BaseBlock(LATENT_SPACE_8, LATENT_SPACE_4, 3, 2, activation=activation, drop_out=drop_out)
        self.conv3 = BaseBlock(LATENT_SPACE_4, LATENT_SPACE_2, 3, 2, activation=activation, drop_out=drop_out)
        self.residual_blocks = nn.Sequential()

        for i in range(0, self.DEPTH_SIZE):
            self.residual_blocks.add_module(str(i),ResidualBlock(LATENT_SPACE_2, LATENT_SPACE_2, stride = 1, activation=activation))

        self.deconv1 = deconv(LATENT_SPACE_2, LATENT_SPACE_4)
        self.norm1 = torch.nn.BatchNorm2d(LATENT_SPACE_4, affine=True)
        self.deconv2 = deconv(LATENT_SPACE_4, LATENT_SPACE_8)
        self.norm2 = torch.nn.BatchNorm2d(LATENT_SPACE_8, affine=True)
        self.refinement = nn.Sequential(
            ConvLayer(LATENT_SPACE_8, LATENT_SPACE_8, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(LATENT_SPACE_8, affine=True),
            activation,
            ConvLayer(LATENT_SPACE_8, LATENT_SPACE_8, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(LATENT_SPACE_8, affine=True),
            activation,
        )
        self.final = ConvLayer(LATENT_SPACE_8, dimension, kernel_size=9, stride=1)
        self.activation = activation

    def forward(self, x):
        skipped = self.conv1(x)
        x = self.conv2(skipped)
        x =self.conv3(x)
        x = self.residual_blocks(x)
        x = self.activation(self.norm1(self.deconv1(x)))
        x = self.activation(self.norm2(self.deconv2(x)))
        x = self.refinement(x + skipped)
        x = self.final(x)
        return torch.tanh(x)


class StanfordStrongGenerator(torch.nn.Module):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation=nn.LeakyReLU(), drop_out : float = 0.5):
        super(StanfordStrongGenerator, self).__init__()
        self.DEPTH_SIZE = int(9)
        self.conv1 = BaseBlock(dimension, 64, 9, 1, activation=activation, drop_out=drop_out)
        self.conv2 = BaseBlock(64, 128, 3, 2, activation=activation, drop_out=drop_out)
        self.conv3 = BaseBlock(128, 256, 3, 2, activation=activation, drop_out=drop_out)
        self.residual_blocks = nn.Sequential()

        for i in range(0, self.DEPTH_SIZE):
            self.residual_blocks.add_module(str(i), ResidualBlock(256, 256, stride=1, activation=activation))

        self.deconv1 = deconv(256, 128)
        self.norm1 = torch.nn.BatchNorm2d(128, affine=True)
        self.deconv2 = deconv(128, 64)
        self.norm2 = torch.nn.BatchNorm2d(64, affine=True)
        self.final = ConvLayer(64, dimension, kernel_size=9, stride=1)
        self.activation = activation

    def forward(self, x):
        skip = self.conv1(x)
        e1 = self.conv2(skip)
        e2 = self.conv3(e1)
        x = self.residual_blocks(e2)
        x = self.activation(self.norm1(self.deconv1(x)))
        x = self.activation(self.norm2(self.deconv2(x)))
        x = self.final(x + skip)
        return torch.tanh(x)


class StanfordModernGenerator(StanfordStrongGenerator):
    def __init__(self, dimension, deconv=UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(StanfordModernGenerator, self).__init__(dimension, deconv, activation, drop_out)
        self.refinement = nn.Sequential(
            BaseBlock(dimension, 64, 3, 1, activation=activation, drop_out=drop_out),
            BaseBlock(dimension, 64, 3, 1, activation=activation, drop_out=drop_out),
        )

    def forward(self, x):
        skip = self.conv1(x)
        x = self.conv2(skip)
        x = self.conv3(x)
        x = self.residual_blocks(x)
        x = self.activation(self.norm4(self.deconv1(x )))
        x = self.activation(self.norm5(self.deconv2(x)))
        x = self.refinement(x + skip)
        x = self.final(x)
        return torch.tanh(x)

class StanfordSupremeGenerator(torch.nn.Module):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation=nn.LeakyReLU(), drop_out : float = 0.5):
        super(StanfordSupremeGenerator, self).__init__()
        self.DEPTH_SIZE = int(9)
        self.conv1 = BaseBlock(dimension, 64, 9, 1, activation=activation, drop_out=drop_out)
        self.conv2 = BaseBlock(64, 128, 3, 2, activation=activation, drop_out=drop_out)
        self.conv3 = BaseBlock(128, 256, 3, 2, activation=activation, drop_out=drop_out)
        self.residual_blocks = nn.Sequential()

        for i in range(0, self.DEPTH_SIZE):
            self.residual_blocks.add_module(str(i), ResidualBlock(256, 256, stride=1, activation=activation))

        self.deconv1 = deconv(512, 128)
        self.norm1 = torch.nn.BatchNorm2d(128, affine=True)
        self.deconv2 = deconv(256, 64)
        self.norm2 = torch.nn.BatchNorm2d(64, affine=True)
        self.final = ConvLayer(64, dimension, kernel_size=9, stride=1)
        self.activation = activation

    def forward(self, x):
        skip = self.conv1(x)
        e1 = self.conv2(skip)
        e2 = self.conv3(e1)
        x = self.residual_blocks(e2)
        x = self.activation(self.norm1(self.deconv1(torch.cat([x, e2], dim=1))))
        x = self.activation(self.norm2(self.deconv2(torch.cat([x, e1], dim=1))))
        x = self.final(x + skip)
        return torch.tanh(x)
