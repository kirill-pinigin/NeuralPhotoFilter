import torch
import torch.nn as nn
from torchvision import models

from torch.nn.parameter import Parameter

FEATURE_OFFSET = int(1)

OXFORD_CONFIG = {'dnn' : models.vgg11(pretrained=True).features, 'features' : [3, 6, 11, 16]}
OXFORD_BN_CONFIG = {'dnn': models.vgg11_bn(pretrained=True).features, 'features' : [4, 8, 15, 22]}

VGG_16_CONFIG = {'dnn' : models.vgg16(pretrained=True).features, 'features' :  [4 - FEATURE_OFFSET, 9 - FEATURE_OFFSET, 16 - FEATURE_OFFSET,  23 - FEATURE_OFFSET]}
VGG_16_BN_CONFIG = {'dnn' : models.vgg16_bn(pretrained=True).features, 'features' :  [6 - FEATURE_OFFSET * 2, 13 - FEATURE_OFFSET * 2, 23 - FEATURE_OFFSET * 2, 33 - FEATURE_OFFSET * 2]}

VGG_19_CONFIG = {'dnn' : models.vgg19(pretrained=True).features, 'features' : [4 - FEATURE_OFFSET,  9 - FEATURE_OFFSET, 18 - FEATURE_OFFSET, 36 - FEATURE_OFFSET]}
VGG_19_BN_CONFIG = {'dnn': models.vgg19_bn(pretrained=True).features, 'features' : [6 - FEATURE_OFFSET * 2, 13 - FEATURE_OFFSET * 2, 23 - FEATURE_OFFSET * 2, 52 - FEATURE_OFFSET * 2]}

TUBINGEN_CONFIG = {'dnn' : models.vgg19(pretrained=True).features, 'features' : [5, 10, 19, 28]}
TUBINGEN_BN_CONFIG = {'dnn': models.vgg19_bn(pretrained=True).features, 'features' : [7 , 14, 27, 40]}

def compute_gram_matrix(x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (h * w * ch)
    return G


class BasicFeatureExtractor(nn.Module):
    def __init__(self, dimension, vgg_config , feature_limit = 9):
        super(BasicFeatureExtractor, self).__init__()
        if dimension == 3:
            self.mean = Parameter(torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
            self.std = Parameter(torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))

        vgg_pretrained = vgg_config['dnn']
        conv = BasicFeatureExtractor.configure_input(dimension, vgg_pretrained)

        self.feat1 = nn.Sequential(conv)
        for x in range(1, feature_limit):
            self.feat1.add_module(str(x), vgg_pretrained[x])

    @staticmethod
    def configure_input(dimension, vgg):
        conv = nn.Conv2d(dimension, 64, kernel_size=3, padding=1)

        if dimension == 1 or dimension == 3:
            weight = torch.FloatTensor(64, dimension, 3, 3)
            parameters = list(vgg.parameters())
            for i in range(64):
                if dimension == 1:
                    weight[i, :, :, :] = parameters[0].data[i].mean(0)
                else:
                    weight[i, :, :, :] = parameters[0].data[i]
            conv.weight.data.copy_(weight)
            conv.bias.data.copy_(parameters[1].data)

        return conv

    def forward(self, x):
        if x.size(1) == 3:
            if self.mean.device != x.device:
                self.mean.to(x.device)

            if self.std.device != x.device:
                self.std.to(x.device)

            x = (x - self.mean) / self.std

        return self.feat1(x)


class BasicMultiFeatureExtractor(BasicFeatureExtractor):
    def __init__(self, dimension, vgg_config , requires_grad):
        super(BasicMultiFeatureExtractor, self).__init__(dimension, vgg_config, vgg_config['features'][0])
        vgg_pretrained = vgg_config['dnn']

        self.feat2 = torch.nn.Sequential()
        for x in range(vgg_config['features'][0], vgg_config['features'][1]):
            self.feat2.add_module(str(x), vgg_pretrained[x])

        self.feat3 = torch.nn.Sequential()
        for x in range(vgg_config['features'][1], vgg_config['features'][2]):
            self.feat3.add_module(str(x), vgg_pretrained[x])

        self.feat4 = torch.nn.Sequential()
        for x in range(vgg_config['features'][2], vgg_config['features'][3]):
            self.feat4.add_module(str(x), vgg_pretrained[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_feat1 = super(BasicMultiFeatureExtractor, self).forward(x)
        h_feat2 = self.feat2(h_feat1)
        h_feat3 = self.feat3(h_feat2)
        h_feat4 = self.feat4(h_feat3)
        return h_feat1, h_feat2, h_feat3, h_feat4


class ChromaEdgeExtractor(nn.Module):
    def __init__(self, dimension):
        super(ChromaEdgeExtractor, self).__init__()
        features = models.vgg19(pretrained=True).features
        conv = nn.Conv2d(dimension, 64, kernel_size=3, padding=1)

        if dimension == 3:
            self.mean = Parameter(torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
            self.std = Parameter(torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))

        if dimension == 1 or dimension == 3:
            weight = torch.FloatTensor(64, dimension, 3, 3)
            parameters = list(features.parameters())
            for i in range(64):
                if dimension == 1:
                    weight[i, :, :, :] = parameters[0].data[i].mean(0)
                else:
                    weight[i, :, :, :] = parameters[0].data[i]
            conv.weight.data.copy_(weight)
            conv.bias.data.copy_(parameters[1].data)

        self.feat1_1 = torch.nn.Sequential()
        self.feat1_2 = torch.nn.Sequential()

        self.feat2_1 = torch.nn.Sequential()
        self.feat2_2 = torch.nn.Sequential()

        self.feat1_1.add_module(str(0), conv)
        self.feat1_1.add_module(str(1), features[1])

        for x in range(2, 4 - FEATURE_OFFSET):
            self.feat1_2.add_module(str(x), features[x])

        for x in range(4 - FEATURE_OFFSET, 7 - FEATURE_OFFSET):
            self.feat2_1.add_module(str(x), features[x])

        for x in range(7 - FEATURE_OFFSET, 9 - FEATURE_OFFSET):
            self.feat2_2.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.size(1) == 3:
            if self.mean.device != x.device:
                self.mean.to(x.device)

            if self.std.device != x.device:
                self.std.to(x.device)

            x = (x - self.mean) / self.std

        feat1_1 = self.feat1_1(x)
        feat1_2 = self.feat1_2(feat1_1)

        feat2_1 = self.feat2_1(feat1_2)
        feat2_2 = self.feat2_2(feat2_1)

        out = {
            'feat1_1': feat1_1,
            'feat1_2': feat1_2,

            'feat2_1': feat2_1,
            'feat2_2': feat2_2,
        }
        return out


class ChromaEdgePerceptualCriterion(nn.Module):
    def __init__(self, dimension, weight:float = 1e-3):
        super(ChromaEdgePerceptualCriterion, self).__init__()
        self.features = ChromaEdgeExtractor(dimension)
        self.features.eval()
        self.weight = weight
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cudas = list(range(torch.cuda.device_count()))
        self.features.to(self.device)
        self.distance = nn.MSELoss()
        self.factors = [1.0, 1.0/2.0, 1.0/3.0, 1.0/4.0]

    def forward(self, actual, desire):
        actuals = self.features(actual)
        desires = self.features(desire)
        content_loss = 0.0
        content_loss += self.factors[0]*self.distance(actuals['feat1_1'], desires['feat1_1'])
        content_loss += self.factors[1]*self.distance(actuals['feat1_2'], desires['feat1_2'])
        content_loss += self.factors[2]*self.distance(actuals['feat2_1'], desires['feat2_1'])
        content_loss += self.factors[3]*self.distance(actuals['feat2_2'], desires['feat2_2'])
        del actuals, desires
        return content_loss


class EchelonExtractor(nn.Module):
    def __init__(self, dimension):
        super(EchelonExtractor, self).__init__()
        features = models.vgg19(pretrained=True).features
        conv = nn.Conv2d(dimension, 64, kernel_size=3, padding=1)

        if dimension == 3:
            self.mean = Parameter(torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
            self.std = Parameter(torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))

        if dimension == 1 or dimension == 3:
            weight = torch.FloatTensor(64, dimension, 3, 3)
            parameters = list(features.parameters())
            for i in range(64):
                if dimension == 1:
                    weight[i, :, :, :] = parameters[0].data[i].mean(0)
                else:
                    weight[i, :, :, :] = parameters[0].data[i]
            conv.weight.data.copy_(weight)
            conv.bias.data.copy_(parameters[1].data)

        self.feat1_1 = torch.nn.Sequential()
        self.feat2_1 = torch.nn.Sequential()
        self.feat3_1 = torch.nn.Sequential()
        self.feat4_1 = torch.nn.Sequential()
        self.feat5_1 = torch.nn.Sequential()

        self.feat1_1.add_module(str(0), conv)
        self.feat1_1.add_module(str(1), features[1])

        for x in range(2, 7 - FEATURE_OFFSET):
            self.feat2_1.add_module(str(x), features[x])

        for x in range(7 - FEATURE_OFFSET, 12 - FEATURE_OFFSET):
            self.feat3_1.add_module(str(x), features[x])

        for x in range(12 - FEATURE_OFFSET, 21 - FEATURE_OFFSET):
            self.feat4_1.add_module(str(x), features[x])

        for x in range(21 - FEATURE_OFFSET, 30 - FEATURE_OFFSET):
            self.feat5_1.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.size(1) == 3:
            if self.mean.device != x.device:
                self.mean.to(x.device)

            if self.std.device != x.device:
                self.std.to(x.device)

            x = (x - self.mean) / self.std

        feat1_1 = self.feat1_1(x)
        feat2_1 = self.feat2_1(feat1_1)
        feat3_1 = self.feat3_1(feat2_1)
        feat4_1 = self.feat4_1(feat3_1)
        feat5_1 = self.feat5_1(feat4_1)

        out = {
            'feat1_1': feat1_1,
            'feat2_1': feat2_1,
            'feat3_1': feat3_1,
            'feat4_1': feat4_1,
            'feat5_1': feat5_1,
        }
        return out


class EchelonPerceptualCriterion(nn.Module):
    def __init__(self, dimension, weight:float = 1e-3):
        super(EchelonPerceptualCriterion, self).__init__()
        self.features = EchelonExtractor(dimension)
        self.features.eval()
        self.weight = weight
        self.distance = nn.L1Loss()
        self.factors = [1.0, 1.0/2.0, 1.0/3.0, 1.0/4.0, 1.0/5.0]

    def forward(self, actual, desire):
        actuals = self.features(actual)
        desires = self.features(desire)
        content_loss = 0.0
        content_loss += self.factors[0] * self.distance(actuals['feat1_1'], desires['feat1_1'])
        content_loss += self.factors[1] * self.distance(actuals['feat2_1'], desires['feat2_1'])
        content_loss += self.factors[2] * self.distance(actuals['feat3_1'], desires['feat3_1'])
        content_loss += self.factors[3] * self.distance(actuals['feat4_1'], desires['feat4_1'])
        content_loss += self.factors[4] * self.distance(actuals['feat5_1'], desires['feat5_1'])
        content_loss += self.weight * self.distance(compute_gram_matrix(actuals['feat4_1']), compute_gram_matrix(desires['feat4_1']))
        del actuals, desires
        return content_loss

'''
Perceptual Losses for Real-Time Style Transfer
and Super-Resolution
Justin Johnson, Alexandre Alahi, Li Fei-Fei
{jcjohns, alahi, feifeili}@cs.stanford.edu
Department of Computer Science, Stanford University
'''

class FastNeuralStyleExtractor(BasicMultiFeatureExtractor):
    def __init__(self, dimension, requires_grad=False , bn = True):
        features = VGG_16_BN_CONFIG if bn else VGG_16_CONFIG
        super(FastNeuralStyleExtractor, self).__init__(dimension, features, requires_grad)


class FastNeuralStylePerceptualCriterion(nn.Module):
    def __init__(self, dimension, weight:float = 1e-2):
        super(FastNeuralStylePerceptualCriterion, self).__init__()
        self.factors = [1.0, 1.0/2.0, 1.0/3.0, 1.0/4.0]
        self.weight = weight
        self.features = FastNeuralStyleExtractor(dimension)
        self.features.eval()
        self.distance = nn.MSELoss()

    def forward(self, actual, desire):
        actuals = self.features(actual)
        desires = self.features(desire)
        loss = 0.0
        for i in range(len(actuals)):
            loss += self.factors[i] * self.distance(actuals[i], desires[i])

        loss += + self.weight*self.distance(compute_gram_matrix(actuals[2]), compute_gram_matrix(desires[2]))
        del actuals, desires
        return loss


class MobileExtractor(BasicMultiFeatureExtractor):
    def __init__(self, dimension, requires_grad=False, bn=True):
        features = VGG_19_BN_CONFIG if bn else VGG_19_CONFIG
        super(MobileExtractor, self).__init__(dimension, features, requires_grad)


class MobilePerceptualCriterion(nn.Module):
    def __init__(self, dimension):
        super(MobilePerceptualCriterion, self).__init__()
        self.factors = [1.0, 1.0/2.0, 1.0/3.0, 1.0/4.0]
        self.features = MobileExtractor(dimension)
        self.features.eval()
        self.distance = nn.MSELoss()

    def forward(self, actual, desire):
        actuals = self.features(actual)
        desires = self.features(desire)
        loss = 0.0
        for i in range(len(actuals)):
            loss += self.factors[i]*self.distance(actuals[i], desires[i])

        del actuals, desires
        return loss

'''
https://www.robots.ox.ac.uk/~vgg/research/very_deep/
'''

class OxfordExtractor(BasicMultiFeatureExtractor):
    def __init__(self, dimension, requires_grad=False, bn=True):
        features = OXFORD_BN_CONFIG if bn else OXFORD_CONFIG
        super(OxfordExtractor, self).__init__(dimension, features, requires_grad)


class OxfordPerceptualCriterion(MobilePerceptualCriterion):
    def __init__(self, dimension):
        super(OxfordPerceptualCriterion, self).__init__(dimension)
        self.features = OxfordExtractor(dimension)
        self.features.eval()


class SharpExtractor(nn.Module):
    def __init__(self, dimension):
        super(SharpExtractor, self).__init__()
        features = models.vgg19(pretrained=True).features
        conv = nn.Conv2d(dimension, 64, kernel_size=3, padding=1)

        if dimension == 3:
            self.mean = Parameter(torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
            self.std = Parameter(torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))

        if dimension == 1 or dimension == 3:
            weight = torch.FloatTensor(64, dimension, 3, 3)
            parameters = list(features.parameters())
            for i in range(64):
                if dimension == 1:
                    weight[i, :, :, :] = parameters[0].data[i].mean(0)
                else:
                    weight[i, :, :, :] = parameters[0].data[i]
            conv.weight.data.copy_(weight)
            conv.bias.data.copy_(parameters[1].data)

        self.feat4_1 = torch.nn.Sequential()
        self.feat4_2 = torch.nn.Sequential()
        self.feat4_3 = torch.nn.Sequential()
        self.feat4_4 = torch.nn.Sequential()
        self.feat5_1 = torch.nn.Sequential()

        self.feat4_1.add_module(str(0), conv)
        self.feat4_1.add_module(str(1), features[1])

        for x in range(2, 21 - FEATURE_OFFSET):
            self.feat4_1.add_module(str(x), features[x])

        for x in range(21 - FEATURE_OFFSET, 23 - FEATURE_OFFSET):
            self.feat4_2.add_module(str(x), features[x])

        for x in range(23 - FEATURE_OFFSET, 25 - FEATURE_OFFSET):
            self.feat4_3.add_module(str(x), features[x])

        for x in range(25 - FEATURE_OFFSET, 27 - FEATURE_OFFSET):
            self.feat4_4.add_module(str(x), features[x])

        for x in range(27 - FEATURE_OFFSET, 30 - FEATURE_OFFSET):
            self.feat5_1.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.size(1) == 3:
            if self.mean.device != x.device:
                self.mean.to(x.device)

            if self.std.device != x.device:
                self.std.to(x.device)

            x = (x - self.mean) / self.std

        feat4_1 = self.feat4_1(x)
        feat4_2 = self.feat4_2(feat4_1)
        feat4_3 = self.feat4_3(feat4_2)
        feat4_4 = self.feat4_4(feat4_3)
        feat5_1 = self.feat5_1(feat4_4)

        out = {
            'feat4_1': feat4_1,
            'feat4_2': feat4_2,
            'feat4_3': feat4_3,
            'feat4_4': feat4_4,
            'feat5_1': feat5_1,
        }
        return out


class SharpPerceptualCriterion(FastNeuralStylePerceptualCriterion):
    def __init__(self , dimension):
        super(SharpPerceptualCriterion, self).__init__(dimension)
        self.features = SharpExtractor(dimension)
        self.features.eval()
        self.distance = nn.MSELoss()

    def forward(self, actual, desire):
        actuals = self.features(actual)
        desires = self.features(desire)
        loss = 0.0
        loss += self.distance(actuals['feat4_1'], desires['feat4_1'])
        loss += self.distance(actuals['feat4_2'], desires['feat4_2'])
        loss += self.distance(actuals['feat4_3'], desires['feat4_3'])
        loss += self.distance(actuals['feat4_4'], desires['feat4_4'])
        loss += self.distance(actuals['feat5_1'], desires['feat5_1'])
        del actuals, desires
        return loss


class SigmaExtractor(nn.Module):
    def __init__(self, dimension):
        super(SigmaExtractor, self).__init__()
        features = models.vgg19(pretrained=True).features
        conv = nn.Conv2d(dimension, 64, kernel_size=3, padding=1)

        if dimension == 3:
            self.mean = Parameter(torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
            self.std = Parameter(torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))

        if dimension == 1 or dimension == 3:
            weight = torch.FloatTensor(64, dimension, 3, 3)
            parameters = list(features.parameters())
            for i in range(64):
                if dimension == 1:
                    weight[i, :, :, :] = parameters[0].data[i].mean(0)
                else:
                    weight[i, :, :, :] = parameters[0].data[i]
            conv.weight.data.copy_(weight)
            conv.bias.data.copy_(parameters[1].data)

        self.slice1 = nn.Sequential(conv)
        self.feat1_1 = torch.nn.Sequential()
        self.feat1_2 = torch.nn.Sequential()

        self.feat2_1 = torch.nn.Sequential()
        self.feat2_2 = torch.nn.Sequential()

        self.feat3_1 = torch.nn.Sequential()
        self.feat3_2 = torch.nn.Sequential()
        self.feat3_3 = torch.nn.Sequential()
        self.feat3_4 = torch.nn.Sequential()

        self.feat4_1 = torch.nn.Sequential()
        self.feat4_2 = torch.nn.Sequential()
        self.feat4_3 = torch.nn.Sequential()
        self.feat4_4 = torch.nn.Sequential()

        self.feat5_1 = torch.nn.Sequential()
        self.feat5_2 = torch.nn.Sequential()
        self.feat5_3 = torch.nn.Sequential()
        self.feat5_4 = torch.nn.Sequential()

        self.feat1_1.add_module(str(0), conv)
        self.feat1_1.add_module(str(1), features[1])

        for x in range(2, 4 - FEATURE_OFFSET):
            self.feat1_2.add_module(str(x), features[x])

        for x in range(4 - FEATURE_OFFSET, 7 - FEATURE_OFFSET):
            self.feat2_1.add_module(str(x), features[x])

        for x in range(7 - FEATURE_OFFSET, 9 - FEATURE_OFFSET):
            self.feat2_2.add_module(str(x), features[x])

        for x in range(9 - FEATURE_OFFSET, 12 - FEATURE_OFFSET):
            self.feat3_1.add_module(str(x), features[x])

        for x in range(12 - FEATURE_OFFSET, 14 - FEATURE_OFFSET):
            self.feat3_2.add_module(str(x), features[x])

        for x in range(14 - FEATURE_OFFSET, 16- FEATURE_OFFSET):
            self.feat3_3.add_module(str(x), features[x])

        for x in range(16 - FEATURE_OFFSET, 18 - FEATURE_OFFSET):
            self.feat3_4.add_module(str(x), features[x])

        for x in range(18 - FEATURE_OFFSET, 21 - FEATURE_OFFSET):
            self.feat4_1.add_module(str(x), features[x])

        for x in range(21 - FEATURE_OFFSET, 23 - FEATURE_OFFSET):
            self.feat4_2.add_module(str(x), features[x])

        for x in range(23 - FEATURE_OFFSET, 25- FEATURE_OFFSET):
            self.feat4_3.add_module(str(x), features[x])

        for x in range(25 - FEATURE_OFFSET, 27 - FEATURE_OFFSET):
            self.feat4_4.add_module(str(x), features[x])

        for x in range(27 - FEATURE_OFFSET, 30 - FEATURE_OFFSET):
            self.feat5_1.add_module(str(x), features[x])

        for x in range(30 - FEATURE_OFFSET, 32 - FEATURE_OFFSET):
            self.feat5_2.add_module(str(x), features[x])

        for x in range(32 - FEATURE_OFFSET, 34 - FEATURE_OFFSET):
            self.feat5_3.add_module(str(x), features[x])

        for x in range(34 - FEATURE_OFFSET, 36 - FEATURE_OFFSET):
            self.feat5_4.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.size(1) == 3:
            if self.mean.device != x.device:
                self.mean.to(x.device)

            if self.std.device != x.device:
                self.std.to(x.device)

            x = (x - self.mean) / self.std

        feat1_1 = self.feat1_1(x)
        feat1_2 = self.feat1_2(feat1_1)

        feat2_1 = self.feat2_1(feat1_2)
        feat2_2 = self.feat2_2(feat2_1)

        feat3_1 = self.feat3_1(feat2_2)
        feat3_2 = self.feat3_2(feat3_1)
        feat3_3 = self.feat3_3(feat3_2)
        feat3_4 = self.feat3_4(feat3_3)

        feat4_1 = self.feat4_1(feat3_4)
        feat4_2 = self.feat4_2(feat4_1)
        feat4_3 = self.feat4_3(feat4_2)
        feat4_4 = self.feat4_4(feat4_3)

        feat5_1 = self.feat5_1(feat4_4)
        feat5_2 = self.feat5_2(feat5_1)
        feat5_3 = self.feat5_3(feat5_2)
        feat5_4 = self.feat5_4(feat5_3)

        out = {
            'feat1_1': feat1_1,
            'feat1_2': feat1_2,

            'feat2_1': feat2_1,
            'feat2_2': feat2_2,

            'feat3_1': feat3_1,
            'feat3_2': feat3_2,
            'feat3_3': feat3_3,
            'feat3_4': feat3_4,

            'feat4_1': feat4_1,
            'feat4_2': feat4_2,
            'feat4_3': feat4_3,
            'feat4_4': feat4_4,

            'feat5_1': feat5_1,
            'feat5_2': feat5_2,
            'feat5_3': feat5_3,
            'feat5_4': feat5_4,
        }
        return out


class SigmaPerceptualCriterion(FastNeuralStylePerceptualCriterion):
    def __init__(self ,dimension, weight:float = 1e-1):
        super(SigmaPerceptualCriterion, self).__init__(dimension, weight)
        self.features = SigmaExtractor(dimension)
        self.features.eval()
        self.distance = nn.MSELoss()

    def forward(self, actual, desire):
        actuals = self.features(actual)
        desires = self.features(desire)
        content_loss = 0.0
        content_loss += self.distance(actuals['feat2_1'], desires['feat2_1'])
        content_loss += self.distance(actuals['feat2_2'], desires['feat2_2'])
        content_loss += self.distance(actuals['feat3_1'], desires['feat3_1'])
        content_loss += self.distance(actuals['feat3_2'], desires['feat3_2'])
        content_loss += self.distance(actuals['feat3_3'], desires['feat3_3'])

        style_loss = 0.0
        if self.weight != 0.0:
            style_loss += self.weight*self.distance(compute_gram_matrix(actuals['feat4_1']), compute_gram_matrix(desires['feat4_1']))
            style_loss += self.weight*self.distance(compute_gram_matrix(actuals['feat4_2']), compute_gram_matrix(desires['feat4_2']))
            style_loss += self.weight*self.distance(compute_gram_matrix(actuals['feat4_3']), compute_gram_matrix(desires['feat4_3']))
            style_loss += self.weight*self.distance(compute_gram_matrix(actuals['feat4_4']), compute_gram_matrix(desires['feat4_4']))
            style_loss += self.weight*self.distance(compute_gram_matrix(actuals['feat5_2']), compute_gram_matrix(desires['feat5_2']))

        del actuals, desires
        return  content_loss + style_loss


class SimpleExtractor(BasicFeatureExtractor):
    def __init__(self, dimension, bn=True):
        features_list = VGG_19_BN_CONFIG['features'] if bn else VGG_19_CONFIG['features']
        features_limit = features_list[1]
        super(SimpleExtractor, self).__init__(dimension, VGG_19_CONFIG, features_limit)


class SimplePerceptualCriterion(nn.Module):
    def __init__(self, dimension):
        super(SimplePerceptualCriterion, self).__init__()
        self.features = SimpleExtractor(dimension)
        self.features.eval()
        self.distance = nn.MSELoss()

    def forward(self, actual, desire):
        actuals = self.features(actual)
        desires = self.features(desire)
        loss = self.distance(actuals, desires)
        del actuals, desires
        return loss

'''
A Neural Algorithm of Artistic Style
Leon A. Gatys,1,2,3∗ Alexander S. Ecker,1,2,4,5 Matthias Bethge1,2,4
1Werner Reichardt Centre for Integrative Neuroscience
and Institute of Theoretical Physics, University of Tubingen, Germany ¨
2Bernstein Center for Computational Neuroscience, Tubingen, Germany ¨
3Graduate School for Neural Information Processing, Tubingen, Germany ¨
4Max Planck Institute for Biological Cybernetics, Tubingen, Germany ¨
5Department of Neuroscience, Baylor College of Medicine, Houston, TX, USA
'''

class TubingenExtractor(BasicMultiFeatureExtractor):
    def __init__(self, dimension, requires_grad=False, bn=True):
        features = TUBINGEN_BN_CONFIG if bn else TUBINGEN_CONFIG
        super(TubingenExtractor, self).__init__(dimension, features, requires_grad)


class TubingenPerceptualCriterion(MobilePerceptualCriterion):
    def __init__(self, dimension):
        super(TubingenPerceptualCriterion, self).__init__(dimension)
        self.features = TubingenExtractor(dimension)
        self.features.eval()
