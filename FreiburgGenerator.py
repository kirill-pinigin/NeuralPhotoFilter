import torch
import torch.nn as nn
from torchvision import models

from NeuralBlocks import BaseBlock, ConvLayer, UpsampleDeConv, AttentionBlock

LATENT_SPACE   = int(64)

'''
U-Net: Convolutional Networks for Biomedical
Image Segmentation
Olaf Ronneberger, Philipp Fischer, and Thomas Brox
Computer Science Department and BIOSS Centre for Biological Signalling Studies,
University of Freiburg, Germany
'''

class FreiburgGenerator(torch.nn.Module):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(FreiburgGenerator, self).__init__()
        self.enc1   = FreiburgDoubleBlock(dimension, 64, activation, drop_out=drop_out)
        self.enc2   = FreiburgDoubleBlock(64,  128, activation, drop_out=drop_out)
        self.enc3   = FreiburgDoubleBlock(128, 256, activation, drop_out=drop_out)
        self.enc4   = FreiburgDoubleBlock(256, 512, activation, drop_out=drop_out)

        self.center = FreiburgDoubleBlock(512, 1024)

        self.deconv4 = deconv(1024, 512)
        self.dec4  = FreiburgSingleBlock(1024, 512, activation = activation, drop_out=drop_out)
        self.deconv3 = deconv(512, 256)
        self.dec3  = FreiburgSingleBlock(512,  256, activation = activation, drop_out=drop_out)
        self.deconv2 = deconv(256, 128)
        self.dec2  = FreiburgSingleBlock(256,  128, activation = activation, drop_out=drop_out)
        self.deconv1 = deconv(128, 64)
        self.dec1  = FreiburgSingleBlock(128,   64, activation = activation, drop_out=drop_out)
        self.final  = ConvLayer(64, dimension, 1)
        self.activation = activation
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.enc1(x)
        pool1 = self.max_pool(e1)
        e2 = self.enc2(pool1)
        pool2 = self.max_pool(e2)
        e3 = self.enc3(pool2)
        pool3 = self.max_pool(e3)
        e4 = self.enc4(pool3)
        pool4 = self.max_pool(e4)

        c = self.center(pool4)

        up6 = self.activation(self.deconv4(c))
        merge6 = torch.cat([up6, e4], dim=1)
        conv6 = self.dec4(merge6)
        up7 = (self.deconv3(conv6))
        merge7 = torch.cat([up7, e3], dim=1)
        conv7 = self.dec3(merge7)
        up8 = self.activation(self.deconv2(conv7))
        merge8 = torch.cat([up8, e2], dim=1)
        conv8 = self.dec2(merge8)
        up9 = self.activation(self.deconv1(conv8))
        merge9 = torch.cat([up9, e1], dim=1)
        conv9 = self.dec1(merge9)
        conv = self.activation(conv9)
        y = self.final(conv)
        return torch.tanh(y)


class FreiburgFastGenerator(FreiburgGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(FreiburgFastGenerator, self).__init__(dimension, deconv, activation, drop_out)
        self.enc1   = FreiburgDoubleBlock(dimension, 32, activation, drop_out=drop_out)
        self.enc2   = FreiburgDoubleBlock(32,  64, activation, drop_out=drop_out)
        self.enc3   = FreiburgDoubleBlock(64, 128, activation, drop_out=drop_out)
        self.enc4   = FreiburgDoubleBlock(128, 256, activation, drop_out=drop_out)

        self.center = FreiburgDoubleBlock(256, 512)

        self.deconv4 = deconv(512, 256)
        self.dec4  = FreiburgSingleBlock(512, 256, activation = activation, drop_out=drop_out)
        self.deconv3 = deconv(256, 128)
        self.dec3  = FreiburgSingleBlock(256,  128, activation = activation, drop_out=drop_out)
        self.deconv2 = deconv(128, 64)
        self.dec2  = FreiburgSingleBlock(128,  64, activation = activation, drop_out=drop_out)
        self.deconv1 = deconv(64, 32)
        self.dec1  = FreiburgSingleBlock(64,   32, activation = activation, drop_out=drop_out)
        self.final  = ConvLayer(32, dimension, 1)


class FreiburgAttentiveGenerator(FreiburgFastGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(FreiburgAttentiveGenerator, self).__init__(dimension, deconv, activation, drop_out)
        self.atb4 = AttentionBlock(256, 128)
        self.atb3 = AttentionBlock(128, 64)
        self.atb2 = AttentionBlock(64, 32)
        self.atb1 = AttentionBlock(32, 16)

    def forward(self, x):
        e1 = self.enc1(x)
        pool1 = self.max_pool(e1)
        e2 = self.enc2(pool1)
        pool2 = self.max_pool(e2)
        e3 = self.enc3(pool2)
        pool3 = self.max_pool(e3)
        e4 = self.enc4(pool3)
        pool4 = self.max_pool(e4)

        c = self.center(pool4)

        d4 = self.activation(self.deconv4(c))
        x4 = self.atb4(g=d4, x=e4)
        merge4 = torch.cat([d4, x4], dim=1)
        x = self.dec4(merge4)
        d3 = (self.deconv3(x))
        x3 = self.atb3(g=d3, x=e3)
        merge3 = torch.cat([d3, x3], dim=1)
        x = self.dec3(merge3)
        d2 = self.activation(self.deconv2(x))
        x2 = self.atb2(g=d2, x=e2)
        merge2 = torch.cat([d2, x2], dim=1)
        x = self.dec2(merge2)
        d1 = self.activation(self.deconv1(x))
        x1 = self.atb1(g=d1, x=e1)
        merge1 = torch.cat([d1, x1], dim=1)
        x = self.dec1(merge1)
        x = self.activation(x)
        x = self.final(x)
        return torch.tanh(x)


class FreiburgModernGenerator(FreiburgAttentiveGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(FreiburgModernGenerator, self).__init__(dimension, deconv, activation, drop_out)

    def forward(self, x):
        return super(FreiburgModernGenerator, self).forward(x) + x


class FreiburgResidualGenerator(nn.Module):
    def __init__(self, dimension, deconv=UpsampleDeConv, activation=nn.LeakyReLU(), drop_out : float = 0.5):
        super(FreiburgResidualGenerator, self).__init__()
        self.activation = activation
        self.max_pool = nn.MaxPool2d(2, 2)
        base_model = models.resnet18(pretrained=True)
        conv = nn.Conv2d(dimension, LATENT_SPACE, kernel_size=7, stride=2, padding=3, bias=False)

        if dimension == 1 or dimension == 3:
            weight = torch.FloatTensor(64, dimension, 7, 7)
            parameters = list(base_model.parameters())
            for i in range(LATENT_SPACE):
                if dimension == 1:
                    weight[i, :, :, :] = parameters[0].data[i].mean(0)
                else:
                    weight[i, :, :, :] = parameters[0].data[i]
            conv.weight.data.copy_(weight)

        self.encoder0 = nn.Sequential(
            conv,
            base_model.bn1,
            activation,
            base_model.maxpool,
        )

        self.encoder1 = base_model.layer1
        self.encoder2 = base_model.layer2
        self.encoder3 = base_model.layer3
        self.encoder4 = base_model.layer4

        self.deconv_center = deconv(512, 512)
        self.center = FreiburgSingleBlock(512, 256, activation=activation, drop_out=drop_out)

        self.deconv5 = deconv(768, 512)
        self.decoder5 = FreiburgSingleBlock(512, 256, activation=activation, drop_out=drop_out)
        self.deconv4 = deconv(512, 512)
        self.decoder4 = FreiburgSingleBlock(512, 256, activation=activation, drop_out=drop_out)
        self.deconv3 = deconv(384, 256)
        self.decoder3 = FreiburgSingleBlock(256, 64, activation=activation, drop_out=drop_out)
        self.deconv2 = deconv(128, 128)
        self.decoder2 = FreiburgSingleBlock(128, 128, activation=activation, drop_out=drop_out)
        self.deconv1 = deconv(128, 128)
        self.decoder1 = FreiburgSingleBlock(128, 32, activation=activation, drop_out=drop_out)
        self.decoder0 = nn.Conv2d(32, 32, 3, padding=1)
        self.final = ConvLayer(32, dimension, 1)
        self.activation = activation

    def freeze(self, flag = True):
        if flag == True:
            for param in self.encoder0:
                param.requires_grad = False

            for param in self.encoder1:
                param.requires_grad = False

            for param in self.encoder2:
                param.requires_grad = False

            for param in self.encoder3:
                param.requires_grad = False

            for param in self.encoder4:
                param.requires_grad = False
        else:
            for param in self.encoder0:
                param.requires_grad = True

            for param in self.encoder1:
                param.requires_grad = True

            for param in self.encoder2:
                param.requires_grad = True

            for param in self.encoder3:
                param.requires_grad = True

            for param in self.encoder4:
                param.requires_grad = True

    def forward(self, x):
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        center = self.center(self.activation(self.deconv_center(self.max_pool(enc4))))

        dec5 = self.decoder5(self.deconv5(torch.cat([center, enc4], dim=1)))
        dec4 = self.decoder4(self.deconv4(torch.cat([dec5, enc3], dim=1)))
        dec3 = self.decoder3(self.deconv3(torch.cat([dec4, enc2], dim=1)))
        dec2 = self.decoder2(self.deconv2(torch.cat([dec3, enc1], dim=1)))
        dec1 = self.decoder1(self.deconv1(dec2))
        dec0 = self.activation(self.decoder0(dec1))

        return torch.tanh(self.final(dec0))


class FreiburgSqueezeGenerator(FreiburgGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(FreiburgSqueezeGenerator, self).__init__(dimension, deconv, activation, drop_out)
        pretrained_features = models.squeezenet1_1(pretrained=True).features
        conv = ConvLayer(dimension, 64, kernel_size=3, stride=1, bias=True)

        if dimension == 1 or dimension == 3:
            weight = torch.FloatTensor(64, dimension, 3, 3)
            parameters = list(pretrained_features.parameters())
            for i in range(64):
                if dimension == 1:
                    weight[i, :, :, :] = parameters[0].data[i].mean(0)
                else:
                    weight[i, :, :, :] = parameters[0].data[i]
            conv.weight.data.copy_(weight)
            conv.bias.data.copy_(parameters[1].data)

        self.enc1 = nn.Sequential(conv)
        for x in range(1, 2):
            self.enc1.add_module(str(x), pretrained_features[x])

        self.enc2 = torch.nn.Sequential()
        self.enc3 = torch.nn.Sequential()
        self.enc4 = torch.nn.Sequential()

        for x in range(3, 5):
            self.enc2.add_module(str(x), pretrained_features[x])
        for x in range(6, 8):
            self.enc3.add_module(str(x), pretrained_features[x])
        for x in range(9, 13):
            self.enc4.add_module(str(x), pretrained_features[x])


class FreiburgSupremeGenerator(FreiburgResidualGenerator):
    def __init__(self, dimension, deconv = UpsampleDeConv, activation = nn.LeakyReLU(), drop_out : float = 0.5):
        super(FreiburgSupremeGenerator, self).__init__(dimension, deconv, activation, drop_out)
        self.DEPTH_SIZE = 3
        self.skip1 = BaseBlock(64, 64, 3, 1, activation=activation, drop_out=drop_out)
        self.skip2 = BaseBlock(128, 128, 3, 1, activation=activation, drop_out=drop_out)
        self.skip3 = BaseBlock(256, 256, 3, 1, activation=activation, drop_out=drop_out)
        self.skip4 = BaseBlock(512, 512, 3, 1, activation=activation, drop_out=drop_out)

    def forward(self, x):
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        center = self.center(self.activation(self.deconv_center(self.max_pool(enc4))))

        dec5 = self.decoder5(self.deconv5(torch.cat([center, self.skip4(enc4)], dim=1)))
        dec4 = self.decoder4(self.deconv4(torch.cat([dec5, self.skip3(enc3)], dim=1)))
        dec3 = self.decoder3(self.deconv3(torch.cat([dec4, self.skip2(enc2)], dim=1)))
        dec2 = self.decoder2(self.deconv2(torch.cat([dec3, self.skip1(enc1)], dim=1)))
        dec1 = self.decoder1(self.deconv1(dec2))
        dec0 = self.activation(self.decoder0(dec1))

        return torch.tanh(self.final(dec0))


class FreiburgTernauGenerator(nn.Module):
    def __init__(self, dimension, deconv=UpsampleDeConv, activation=nn.LeakyReLU(), drop_out : float = 0.5):
        super(FreiburgTernauGenerator, self).__init__()
        self.activation = activation
        base_model = models.vgg11(pretrained=True).features
        conv = nn.Conv2d(dimension, LATENT_SPACE, kernel_size=3, padding=1)

        if dimension == 1 or dimension == 3:
            weight = torch.FloatTensor(64, dimension, 3, 3)
            parameters = list(base_model.parameters())
            for i in range(LATENT_SPACE):
                if dimension == 1:
                    weight[i, :, :, :] = parameters[0].data[i].mean(0)
                else:
                    weight[i, :, :, :] = parameters[0].data[i]
            conv.weight.data.copy_(weight)
            conv.bias.data.copy_(parameters[1].data)

        self.encoder1 = nn.Sequential(conv)
        for x in range(1, 3):
            self.encoder1.add_module(str(x), base_model[x])

        self.encoder2 = nn.Sequential()
        for x in range(3, 6):
            self.encoder2.add_module(str(x), base_model[x])

        self.encoder3 = nn.Sequential()
        for x in range(6, 11):
            self.encoder3.add_module(str(x), base_model[x])

        self.encoder4 = nn.Sequential()
        for x in range(11, 16):
            self.encoder4.add_module(str(x), base_model[x])

        self.encoder5 = nn.Sequential()
        for x in range(16, 21):
            self.encoder5.add_module(str(x), base_model[x])

        self.deconv1= deconv(512, 512)
        self.center = FreiburgSingleBlock(512, 256, activation=activation, drop_out=drop_out)

        self.deconv5 = deconv(768, 512)
        self.decoder5 = FreiburgSingleBlock(512, 256, activation=activation, drop_out=drop_out)
        self.deconv4 = deconv(768, 512)
        self.decoder4 = FreiburgSingleBlock(512, 128, activation=activation, drop_out=drop_out)
        self.deconv3 = deconv(384, 256)
        self.decoder3 = FreiburgSingleBlock(256, 64, activation=activation, drop_out=drop_out)
        self.deconv2 = deconv(192, 128)
        self.decoder2 = FreiburgSingleBlock(128, 32, activation=activation, drop_out=drop_out)
        self.decoder1 = nn.Conv2d(96, 32, 3, padding=1)

        self.final = ConvLayer(32, dimension, 1)
        self.activation = activation

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        center = self.encoder5(enc4)

        center = self.center(self.activation((center)))
        print(center.shape)
        print(center.shape)
        dec5 = self.decoder5(self.deconv5(torch.cat([center, center], dim=1)))

        dec4 = self.decoder4(self.deconv4(torch.cat([dec5, enc4], dim=1)))
        dec3 = self.decoder3(self.deconv3(torch.cat([dec4, enc3], dim=1)))
        dec2 = self.decoder2(self.deconv2(torch.cat([dec3, enc2], dim=1)))
        dec1 = self.decoder1(            (torch.cat([dec2, enc1], dim=1)))

        return torch.tanh(self.final(self.activation(dec1)))


class FreiburgDoubleBlock(nn.Sequential):
    def __init__(self, in_size, out_size, activation=nn.LeakyReLU(0.2), drop_out : float = 0.5):
        super(FreiburgDoubleBlock, self).__init__(
            BaseBlock(in_size,  out_size, 3, 1, activation=activation, drop_out=drop_out),
            BaseBlock(out_size, out_size, 3, 1, activation=activation, drop_out=drop_out),
        )


class FreiburgSingleBlock(nn.Sequential):
    def __init__(self, in_size, out_size, activation=nn.LeakyReLU(0.2), drop_out : float = 0.5):
        super(FreiburgSingleBlock, self).__init__(
            activation,
            BaseBlock(in_size, out_size, 3, 1, activation=activation, drop_out=drop_out),
        )
