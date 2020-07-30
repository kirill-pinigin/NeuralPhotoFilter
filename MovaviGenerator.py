import torch
import torch.nn as nn

from NeuralBlocks import BaseBlock, ConvLayer, ResidualBlock, UpsampleDeConv


class MovaviGenerator(torch.nn.Module):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU()):
        super(MovaviGenerator, self).__init__()
        self.deconv1 = deconv
        self.activation = activation
        self.enc1   = BaseBlock(dimension, 16, 3, 2, activation=activation)
        self.enc2   = nn.Sequential(BaseBlock(16,  16, 3, 1, activation=activation), BaseBlock(16,  16, 3, 1, activation=activation), ConvLayer(16,   32, 3, 2))
        self.enc3   = nn.Sequential(BaseBlock(32,  32, 3, 1, activation=activation), BaseBlock(32,  32, 3, 1, activation=activation), ConvLayer(32,   64, 3, 2))
        self.enc4   = nn.Sequential(BaseBlock(64,  64, 3, 1, activation=activation), BaseBlock(64,  64, 3, 1, activation=activation), ConvLayer(64,  128, 3, 2))
        self.enc5   = nn.Sequential(BaseBlock(128,128, 3, 1, activation=activation), BaseBlock(128,128, 3, 1, activation=activation), ConvLayer(128, 128, 3, 2))
        self.center = nn.Sequential(BaseBlock(128,128, 3, 1, activation=activation), BaseBlock(128,128, 3, 1, activation=activation))
        self.dec6   = nn.Sequential(BaseBlock(256,128, 3, 1, activation=activation), BaseBlock(128,128, 3, 1, activation=activation), deconv(128, 64))
        self.dec7   = nn.Sequential(BaseBlock(192, 64, 3, 1, activation=activation), BaseBlock(64,  64, 3, 1, activation=activation), deconv(64,  32))
        self.dec8   = nn.Sequential(BaseBlock(96,  32, 3, 1, activation=activation), BaseBlock(32,  32, 3, 1, activation=activation), deconv(32,  16))
        self.dec9   = nn.Sequential(BaseBlock(48,  16, 3, 1, activation=activation), BaseBlock(16,  16, 3, 1, activation=activation), deconv(16,  16))
        self.dec10 = deconv(16, 16)
        self.final = ConvLayer(16, dimension, 3, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        c = self.center(enc5)
        up6 = torch.cat([c, enc5], dim = 1)
        dec6 = self.dec6(up6)
        up7 = torch.cat([dec6, enc4], dim=1)
        dec7 = self.dec7(up7)
        up8 = torch.cat([dec7, enc3], dim=1)
        dec8 = self.dec8(up8)
        up9 = torch.cat([dec8, enc2], dim=1)
        dec9 = torch.mul(self.dec9(up9), enc1)
        dec10 = self.dec10(dec9)
        out = self.final(dec10)
        return torch.tanh(out)


class MovaviFastGenerator(MovaviGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU()):
        super(MovaviFastGenerator, self).__init__(dimension, deconv, activation)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        c = self.center(enc5)
        up6 = torch.cat([c, enc5], dim = 1)
        dec6 = self.dec6(up6)
        up7 = torch.cat([dec6, enc4], dim=1)
        dec7 = self.dec7(up7)
        up8 = torch.cat([dec7, enc3], dim=1)
        dec8 = self.dec8(up8)
        up9 = torch.cat([dec8, enc2], dim=1)
        dec9 = torch.add(self.dec9(up9), enc1)
        dec10 = self.dec10(dec9)
        out = self.final(dec10)
        return torch.tanh(out)


class MovaviResidualGenerator(MovaviGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU()):
        super(MovaviResidualGenerator, self).__init__(dimension, deconv, activation)
        self.enc1 = ResidualBlock(dimension, 16,  activation = activation, stride=2)
        self.enc2 = ResidualBlock(16, 32, activation = activation, stride=2)
        self.enc3 = ResidualBlock(32, 64, activation = activation, stride=2)
        self.enc4 = ResidualBlock(64, 128, activation = activation, stride=2)
        self.enc5 = ResidualBlock(128, 128,  activation = activation, stride=2)
        self.center = ResidualBlock(128, 128,  activation = activation, stride=1)


class MovaviStrongGenerator(MovaviResidualGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU()):
        super(MovaviStrongGenerator, self).__init__(dimension, deconv, activation)
        self.enc1 = ResidualBlock(dimension, 32, activation=activation, stride=2)
        self.enc2 = ResidualBlock(32, 64, activation=activation, stride=2)
        self.enc3 = ResidualBlock(64, 128, activation=activation, stride=2)
        self.enc4 = ResidualBlock(128, 256, activation=activation, stride=2)
        self.enc5 = ResidualBlock(256, 256, activation=activation, stride=2)
        self.center = ResidualBlock(256, 256, activation=activation, stride=1)

        self.dec6 = nn.Sequential(BaseBlock(512, 256, 3, 1, activation=activation),
                                  BaseBlock(256, 256, 3, 1, activation=activation), deconv(256, 128))
        self.dec7 = nn.Sequential(BaseBlock(384, 128, 3, 1, activation=activation),
                                  BaseBlock(128, 128, 3, 1, activation=activation), deconv(128, 64))
        self.dec8 = nn.Sequential(BaseBlock(192, 64, 3, 1, activation=activation),
                                  BaseBlock(64, 64, 3, 1, activation=activation), deconv(64, 32))
        self.dec9 = nn.Sequential(BaseBlock(96, 32, 3, 1, activation=activation),
                                  BaseBlock(32, 32, 3, 1, activation=activation), deconv(32, 32))
        self.dec10 = deconv(32, 32)
        self.final = ConvLayer(32, dimension, 3, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        c = self.center(enc5)
        up6 = torch.cat([c,    enc5], dim=1)
        dec6 = self.dec6(up6)
        up7 = torch.cat([dec6, enc4], dim=1)
        dec7 = self.dec7(up7)
        up8 = torch.cat([dec7, enc3], dim=1)
        dec8 = self.dec8(up8)
        up9 = torch.cat([dec8, enc2], dim=1)
        dec9 = self.dec9(up9) * enc1
        dec10 = self.dec10(dec9)
        out = self.final(dec10)
        return torch.tanh(out)


class MovaviSupremeGenerator(MovaviStrongGenerator):
    def __init__(self,dimension,  deconv = UpsampleDeConv, activation = nn.LeakyReLU()):
        super(MovaviSupremeGenerator, self).__init__(dimension, deconv, activation)

        self.skip1 = BaseBlock(64, 64, 3, 1, activation=activation)
        self.skip2 = BaseBlock(128, 128, 3, 1, activation=activation)
        self.skip3 = BaseBlock(256, 256, 3, 1, activation=activation)
        self.skip4 = BaseBlock(256, 256, 3, 1, activation=activation)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        c = self.center(enc5)
        up6 = torch.cat([c, self.skip4(enc5)],    dim = 1)
        dec6 = self.dec6(up6)
        up7 = torch.cat([dec6, self.skip3(enc4)], dim=1)
        dec7 = self.dec7(up7)
        up8 = torch.cat([dec7, self.skip2(enc3)], dim=1)
        dec8 = self.dec8(up8)
        up9 = torch.cat([dec8, self.skip1(enc2)], dim=1)
        dec9 = self.dec9(up9) * enc1
        dec10 = self.dec10(dec9)
        out = self.final(dec10)
        return torch.tanh(out)
