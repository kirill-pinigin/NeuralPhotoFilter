import torch
import torch.nn as nn

from NeuralBlocks import ResidualBlock, ConvLayer, UpsampleDeConv

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
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU()):
        super(StanfordGenerator, self).__init__()
        self.DEPTH_SIZE = int(5)
        self.conv1 = ConvLayer(dimension, LATENT_SPACE_4, kernel_size=9, stride=1)
        self.norm1 = torch.nn.BatchNorm2d(LATENT_SPACE_4, affine=True)
        self.conv2 = ConvLayer(LATENT_SPACE_4, LATENT_SPACE_2, kernel_size=3, stride=2)
        self.norm2 = torch.nn.BatchNorm2d(LATENT_SPACE_2, affine=True)
        self.conv3 = ConvLayer(LATENT_SPACE_2, LATENT_SPACE, kernel_size=3, stride=2)
        self.norm3 = torch.nn.BatchNorm2d(LATENT_SPACE, affine=True)
        self.residual_blocks = nn.Sequential()

        for i in range(0, self.DEPTH_SIZE):
            self.residual_blocks.add_module(str(i),ResidualBlock(LATENT_SPACE, LATENT_SPACE, stride=1, activation=activation))

        self.deconv1 = deconv(LATENT_SPACE, LATENT_SPACE_2)
        self.norm4 = torch.nn.BatchNorm2d(LATENT_SPACE_2, affine=True)
        self.deconv2 = deconv(LATENT_SPACE_2, LATENT_SPACE_4)
        self.norm5 = torch.nn.BatchNorm2d(LATENT_SPACE_4, affine=True)
        self.final = ConvLayer(LATENT_SPACE_4, dimension, kernel_size=9, stride=1)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.activation(self.norm3(self.conv3(x)))
        x = self.residual_blocks(x)
        x = self.activation(self.norm4(self.deconv1(x)))
        x = self.activation(self.norm5(self.deconv2(x)))
        x = self.final(x)
        return torch.tanh(x)


class StanfordFastGenerator(torch.nn.Module):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU()):
        super(StanfordFastGenerator, self).__init__()
        self.DEPTH_SIZE = int(3)
        self.conv1 = ConvLayer(dimension, LATENT_SPACE_8, kernel_size=9, stride=1)
        self.norm1 = torch.nn.BatchNorm2d(LATENT_SPACE_8, affine=True)
        self.conv2 = ConvLayer(LATENT_SPACE_8, LATENT_SPACE_4, kernel_size=3, stride=2)
        self.norm2 = torch.nn.BatchNorm2d(LATENT_SPACE_4, affine=True)
        self.conv3 = ConvLayer(LATENT_SPACE_4, LATENT_SPACE_2, kernel_size=3, stride=2)
        self.norm3 = torch.nn.BatchNorm2d(LATENT_SPACE_2, affine=True)
        self.residual_blocks = nn.Sequential()

        for i in range(0, self.DEPTH_SIZE):
            self.residual_blocks.add_module(str(i),ResidualBlock(LATENT_SPACE_2, LATENT_SPACE_2, stride = 1, activation=activation))

        self.deconv1 = deconv(LATENT_SPACE_2, LATENT_SPACE_4)
        self.norm4 = torch.nn.BatchNorm2d(LATENT_SPACE_4, affine=True)
        self.deconv2 = deconv(LATENT_SPACE_4, LATENT_SPACE_8)
        self.norm5 = torch.nn.BatchNorm2d(LATENT_SPACE_8, affine=True)
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
        skipped = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(skipped)))
        x = self.activation(self.norm3(self.conv3(x)))
        x = self.residual_blocks(x)
        x = self.activation(self.norm4(self.deconv1(x)))
        x = self.activation(self.norm5(self.deconv2(x)))
        x = self.refinement(x + skipped)
        x = self.final(x)
        return torch.tanh(x)


class StanfordStrongGenerator(torch.nn.Module):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation=nn.LeakyReLU()):
        super(StanfordStrongGenerator, self).__init__()
        self.DEPTH_SIZE = int(9)
        self.conv1 = ConvLayer(dimension, 64, kernel_size=9, stride=1)
        self.norm1 = torch.nn.BatchNorm2d(64, affine=True)
        self.conv2 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.norm2 = torch.nn.BatchNorm2d(128, affine=True)
        self.conv3 = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.norm3 = torch.nn.BatchNorm2d(256, affine=True)
        self.residual_blocks = nn.Sequential()

        for i in range(0, self.DEPTH_SIZE):
            self.residual_blocks.add_module(str(i), ResidualBlock(256, 256, stride=1, activation=activation))

        self.deconv1 = deconv(256, 128)
        self.norm4 = torch.nn.BatchNorm2d(128, affine=True)
        self.deconv2 = deconv(128, 64)
        self.norm5 = torch.nn.BatchNorm2d(64, affine=True)
        self.final = ConvLayer(64, dimension, kernel_size=9, stride=1)
        self.activation = activation

    def forward(self, x):
        skip = self.activation(self.norm1(self.conv1(x)))
        e1 = self.activation(self.norm2(self.conv2(skip)))
        e2 = self.activation(self.norm3(self.conv3(e1)))
        x = self.residual_blocks(e2)
        x = self.activation(self.norm4(self.deconv1(x)))
        x = self.activation(self.norm5(self.deconv2(x)))
        x = self.final(x + skip)
        return torch.tanh(x)


class StanfordModernGenerator(StanfordStrongGenerator):
    def __init__(self, dimension, deconv=UpsampleDeConv, activation = nn.LeakyReLU()):
        super(StanfordModernGenerator, self).__init__(dimension, deconv, activation)
        self.refinement = nn.Sequential(
            ConvLayer(64, 64, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(64, affine=True),
            activation,
            ConvLayer(64, 64, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(64, affine=True),
            activation,
        )

    def forward(self, x):
        skip = self.activation(self.norm1(self.conv1(x)))
        x = self.conv2(skip)
        x = self.conv3(x)
        x = self.residual_blocks(x)
        x = self.activation(self.norm4(self.deconv1(x )))
        x = self.activation(self.norm5(self.deconv2(x)))
        x = self.refinement(x + skip)
        x = self.final(x)
        return torch.tanh(x)


class StanfordSupremeGenerator(torch.nn.Module):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation=nn.LeakyReLU()):
        super(StanfordSupremeGenerator, self).__init__()
        self.DEPTH_SIZE = int(9)
        self.conv1 = ConvLayer(dimension, 64, kernel_size=9, stride=1)
        self.norm1 = torch.nn.BatchNorm2d(64, affine=True)
        self.conv2 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.norm2 = torch.nn.BatchNorm2d(128, affine=True)
        self.conv3 = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.norm3 = torch.nn.BatchNorm2d(256, affine=True)
        self.residual_blocks = nn.Sequential()

        for i in range(0, self.DEPTH_SIZE):
            self.residual_blocks.add_module(str(i), ResidualBlock(256, 256, stride=1, activation=activation))

        self.deconv1 = deconv(512, 128)
        self.norm4 = torch.nn.BatchNorm2d(128, affine=True)
        self.deconv2 = deconv(256, 64)
        self.norm5 = torch.nn.BatchNorm2d(64, affine=True)
        self.final = ConvLayer(64, dimension, kernel_size=9, stride=1)
        self.activation = activation

    def forward(self, x):
        skip = self.activation(self.norm1(self.conv1(x)))
        e1 = self.activation(self.norm2(self.conv2(skip)))
        e2 = self.activation(self.norm3(self.conv3(e1)))
        x = self.residual_blocks(e2)
        x = self.activation(self.norm4(self.deconv1(torch.cat([x, e2], dim=1))))
        x = self.activation(self.norm5(self.deconv2(torch.cat([x, e1], dim=1))))
        x = self.final(x + skip)
        return torch.tanh(x)
