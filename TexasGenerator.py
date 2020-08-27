import torch
import torch.nn as nn
from torchvision import models

from NeuralBlocks import BaseBlock, UpsampleDeConv, ConvLayer

LATENT_SPACE = 64

'''DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and BetterOrest 
Kupyn1, 3, Tetiana Martyniuk1, Junru Wu2, Zhangyang Wang21Ukrainian Catholic University, Lviv, Ukraine;3SoftServe, 
Lviv, Ukraine{kupyn, t.martynyuk}@ucu.edu.ua2Department of Computer Science and Engineering, 
Texas A&M University{sandboxmaster, atlaswang}@tamu.edu

with change of backbone because mobile is not export to ONNX
'''

class TexasResidualGenerator(nn.Module):
    def __init__(self, dimension, deconv=UpsampleDeConv, activation=nn.LeakyReLU()):
        super(TexasResidualGenerator, self).__init__()
        self.fpn = ResidualFPN(dimension=dimension, activation = activation)
        self.head1 = nn.Sequential(ConvLayer(LATENT_SPACE, LATENT_SPACE, kernel_size=3, bias=False), activation,
                                   ConvLayer(LATENT_SPACE, LATENT_SPACE, kernel_size=3, bias=False), activation)
        self.head2 = nn.Sequential(ConvLayer(LATENT_SPACE, LATENT_SPACE, kernel_size=3, bias=False), activation,
                                   ConvLayer(LATENT_SPACE, LATENT_SPACE, kernel_size=3, bias=False), activation)
        self.head3 = nn.Sequential(ConvLayer(LATENT_SPACE, LATENT_SPACE, kernel_size=3, bias=False), activation,
                                   ConvLayer(LATENT_SPACE, LATENT_SPACE, kernel_size=3, bias=False), activation)
        self.head4 = nn.Sequential(ConvLayer(LATENT_SPACE, LATENT_SPACE, kernel_size=3, bias=False), activation,
                                   ConvLayer(LATENT_SPACE, LATENT_SPACE, kernel_size=3, bias=False), activation)
        self.smooth1 = BaseBlock(4 * LATENT_SPACE, LATENT_SPACE, kernel_size=3, stride=1, bias=True, activation=activation)
        self.smooth2 = BaseBlock(1 * LATENT_SPACE, LATENT_SPACE, kernel_size=3, stride= 1, bias=True, activation=activation)
        self.deconv1 = deconv(LATENT_SPACE, dimension)
        self.activation = activation

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)
        map4 = nn.functional.interpolate(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.interpolate(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.interpolate(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.interpolate(self.head1(map1), scale_factor=1, mode="nearest")
        smoothed = self.smooth1(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.interpolate(smoothed, scale_factor=2, mode="nearest")
        final = self.deconv1(smoothed)
        return torch.tanh(final) + x


class ResidualFPN(nn.Module):
    def __init__(self, dimension, activation):
        super(ResidualFPN, self).__init__()
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

        self.td1 = BaseBlock(LATENT_SPACE, LATENT_SPACE, kernel_size=3, stride= 1, bias=True, activation=activation)
        self.td2 = BaseBlock(LATENT_SPACE, LATENT_SPACE, kernel_size=3, stride= 1, bias=True, activation=activation)
        self.td3 = BaseBlock(LATENT_SPACE, LATENT_SPACE, kernel_size=3, stride= 1, bias=True, activation=activation)

        self.lateral4 = ConvLayer(512, LATENT_SPACE, kernel_size=1, bias=False)
        self.lateral3 = ConvLayer(256, LATENT_SPACE, kernel_size=1, bias=False)
        self.lateral2 = ConvLayer(128, LATENT_SPACE, kernel_size=1, bias=False)
        self.lateral1 = ConvLayer(64,  LATENT_SPACE, kernel_size=1, bias=False)
        self.lateral0 = ConvLayer(64,  LATENT_SPACE, kernel_size=1, bias=False)

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.interpolate(map4, scale_factor=2, mode="nearest"))
        map2 = self.td2(lateral2 + nn.functional.interpolate(map3, scale_factor=2, mode="nearest"))
        map1 = self.td3(lateral1 + nn.functional.interpolate(map2, scale_factor=2, mode="nearest"))
        map0 = lateral0
        return map0, map1, map2, map3, map4
