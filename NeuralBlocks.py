import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ColorFullnesCriterion(torch.nn.Module):
    def __init__(self):
        super(ColorFullnesCriterion, self).__init__()

    def colorfulness(self, x):
        assert(x.size(1) == 3)
        result = torch.zeros(x.size(0)).to(x.device)
        for i in range(x.size(0)):
            (R, G, B) = x[i][0], x[i][1], x[i][2],
            rg = torch.abs(R - G)
            yb = torch.abs(0.5 * (R + G)- B)
            rgMean, rgStd = torch.mean(rg), torch.std(rg)
            ybMean, ybStd = torch.mean(yb), torch.std(yb)
            stdRoot = torch.sqrt((rgStd ** 2) + (ybStd ** 2))
            meanRoot = torch.sqrt((rgMean ** 2) + (ybMean ** 2))
            result[i] = stdRoot + (0.3 * meanRoot)

        return result

    def forward(self, x, y):
        return nn.functional.l1_loss(self.colorfulness(x), self.colorfulness(y))


class ColorHistCriterion(torch.nn.Module):
    def __init__(self):
        super(ColorHistCriterion, self).__init__()

    def histogram(self, x):
        assert(x.size(1) == 3)
        result = torch.zeros(x.size(0), x.size(1), 255).to(x.device)

        for i in range(x.size(0)):
            R, G, B = torch.round(x[i][0]*255.0), torch.round(x[i][1]*255.0), torch.round(x[i][2]*255.0)
            R, G, B = R.data.cpu().numpy(), G.data.cpu().numpy(), B.data.cpu().numpy()
            rh = np.histogram(R, bins=255)[0]
            gh = np.histogram(G, bins=255)[0]
            bh = np.histogram(B, bins=255)[0]
            result[i][0] = torch.from_numpy(gh).float().view(-1).to(x.device)
            result[i][1] = torch.from_numpy(bh).float().view(-1).to(x.device)
            result[i][2] = torch.from_numpy(rh).float().view(-1).to(x.device)

        return result

    def forward(self, x, y):
        return nn.functional.l1_loss(self.histogram(x), self.histogram(x))


class HueSaturationValueCriterion(torch.nn.Module):
    def __init__(self):
        super(HueSaturationValueCriterion, self).__init__()
        self.criterion = nn.L1Loss()
        self.eps= 1e-6

    def hsv(self, im):
        assert (im.size(1) == 3)
        img = im * 0.5 + 0.5
        hue = torch.Tensor(im.shape[0], im.shape[2], im.shape[3]).to(im.device)

        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6

        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6.0

        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
        saturation[ img.max(1)[0]==0 ] = 0.0

        value = img.max(1)[0]

        return torch.cat((hue, saturation, value), dim=1)

    def forward(self, x, y):
        x_hsv = self.hsv(x)
        y_hsv = self.hsv(y)
        return nn.functional.l1_loss(x_hsv, y_hsv)


class SILU(torch.nn.Module):
    def __init__(self):
        super(SILU, self).__init__()

    def forward(self, x):
        out = torch.mul(x, torch.sigmoid(x))
        return out


class Perceptron(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Padding(torch.nn.Module):
    def __init__(self, padding_size = 1):
        super(Padding, self).__init__()
        self.pad = torch.nn.ReflectionPad2d(padding_size)

    def forward(self, x):
        return self.pad(x)


class ConvLayer(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias = False):
        super(ConvLayer, self).__init__(in_channels, out_channels, kernel_size, stride=stride, bias = bias)
        padding_size = kernel_size // 2
        self.pad = Padding(padding_size)

    def forward(self, x):
        x = self.pad(x)
        x = super(ConvLayer, self).forward(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels, gate_dimension):
        super(AttentionBlock, self).__init__()

        self.tract = nn.Sequential(
            ConvLayer(channels, gate_dimension, kernel_size=1, stride=1,  bias=True),
            nn.BatchNorm2d(gate_dimension)
        )

        self.skip = nn.Sequential(
            ConvLayer(channels, gate_dimension, kernel_size=1, stride=1,  bias=True),
            nn.BatchNorm2d(gate_dimension)
        )

        self.gate = nn.Sequential(
            ConvLayer(gate_dimension, channels, kernel_size=1, stride=1,  bias=True),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        t = self.tract(g)
        s = self.skip(x)
        psi = self.relu(torch.add(t,s))
        psi = self.gate(psi)
        return torch.mul(x, psi)

class BaseBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation = Identity(), bias = False ):
        super(BaseBlock, self).__init__()
        self.model = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride, bias),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.Dropout(),
            activation,
        )

    def forward(self, x):
        return self.model(x)


class UpsampleDeConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels,):
        super(UpsampleDeConv, self).__init__()
        self.conv2d = ConvLayer(in_channels, out_channels, 3, 1, bias=False)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, mode='nearest', scale_factor=2)
        x  = self.conv2d(x)
        return x


class TransposedDeConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransposedDeConv, self).__init__()
        self.conv2d = torch.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

    def forward(self, x):
        return self.conv2d(x)


class PixelDeConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PixelDeConv, self).__init__()
        self.conv2d = ConvLayer(in_channels, out_channels * 4, 3, 1)
        self.upsample = nn.PixelShuffle(2)

    def forward(self, x):
        return self.upsample(self.conv2d(x))


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, activation = nn.LeakyReLU(0.2)):
        super(ResidualBlock, self).__init__()
        self.conv1 = BaseBlock(in_channels,  out_channels, kernel_size=3, stride=stride, activation = activation)
        self.conv2 = BaseBlock(out_channels, out_channels, kernel_size=3, stride=1)
        self.skip  = BaseBlock(in_channels,  out_channels, kernel_size=1, stride=stride, bias= False)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return torch.add(x, residual)


class SimpleEncoder(nn.Module):
    def __init__(self, in_size, out_size, activation=nn.LeakyReLU(0.2), bn = True):
        super(SimpleEncoder, self).__init__()
        layers = [ConvLayer(in_size, out_size, 3, 2)]

        if bn:
            layers +=[nn.BatchNorm2d(out_size)]

        layers +=[activation]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, in_size, out_size, deconv= UpsampleDeConv):
        super(SimpleDecoder, self).__init__()
        layers = [deconv(in_size, out_size),
                  nn.BatchNorm2d(out_size)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class TotalVariation(nn.Module):
    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
