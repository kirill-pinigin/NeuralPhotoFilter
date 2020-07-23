import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import random

from NeuralBlocks import  SpectralNorm, Flatten
from PerceptualCriterion import SQUEEZENET_CONFIG, BasicMultiFeatureExtractor

def compute_gram_matrix(x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (h * w * ch)
    return G


class PerceptualDiscriminator(nn.Module):
    def __init__(self, dimension):
        super(PerceptualDiscriminator, self).__init__()

        self.feat1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=dimension, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )

        self.feat2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.feat3 = torch.nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.feat4 = torch.nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.predictor = torch.nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
            Flatten(),
        )

    def forward(self, x):
        feat1 = self.feat1(x)
        feat2 = self.feat2(feat1)
        feat3 = self.feat3(feat2)
        feat4 = self.feat4(feat3)
        result = self.predictor(feat4)
        feats = [feat1, feat2, feat3, feat4]
        return feats, result

class AdaptivePerceptualCriterion(nn.Module):
    def __init__(self, dimension):
        super(AdaptivePerceptualCriterion, self).__init__()
        self.discriminator = PerceptualDiscriminator(dimension)
        self.distance = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.relu = nn.ReLU()
        self.margin = 1.0
        self.lossG = None
        self.lossD = None
        self.factors = [1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0]

    def forward(self, actual, desire):
        return self.evaluate(actual, desire), self.update(actual, desire)

    def backward(self, retain_variables=True):
        return self.lossG.backward(retain_variables=retain_variables)

    def featurize(self, actual, desire):
        actual_features, a = self.discriminator(actual)
        desire_features, d = self.discriminator(desire)
        ploss = 0.0

        for i in range(len(desire_features)):
            ploss += self.factors[i]*self.distance(actual_features[i], desire_features[i])

        return ploss, a, d

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        ploss, rest, _ = self.featurize(actual, desire)
        ones = Variable(torch.ones(rest.shape).to(actual.device))
        aloss = self.bce(rest, ones)
        self.lossG = ploss + aloss + self.distance(actual, desire)
        return self.lossG

    def update(self, actual, desire):
        self.discriminator.train()
        ploss, fake, real = self.featurize(actual, desire)
        zeros = Variable(torch.zeros(fake.shape).to(actual.device))
        ones = Variable(torch.ones(real.shape).to(actual.device))
        self.lossD = self.bce(real, ones) + self.bce(fake, zeros) + self.relu(self.margin - ploss).mean()
        return self.lossD


class ResidualDiscriminator(PerceptualDiscriminator):
    def __init__(self, dimension):
        super(ResidualDiscriminator, self).__init__()
        base_model = models.resnet18(pretrained=True)
        conv = nn.Conv2d(dimension, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if dimension == 1 or dimension == 3:
            weight = torch.FloatTensor(64, dimension, 7, 7)
            parameters = list(base_model.parameters())
            for i in range(64):
                if dimension == 1:
                    weight[i, :, :, :] = parameters[0].data[i].mean(0)
                else:
                    weight[i, :, :, :] = parameters[0].data[i]
            conv.weight.data.copy_(weight)

        self.encoder0 = nn.Sequential(
            conv,
            base_model.bn1,
            nn.ReLU(),
            base_model.maxpool,
        )

        self.encoder1 = base_model.layer1
        self.encoder2 = base_model.layer2
        self.encoder3 = base_model.layer3
        self.encoder4 = base_model.layer4

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        result = self.predictor(enc4)
        return enc1, enc2, enc3, enc4, result


class ResidualAdaptivePerceptualCriterion(AdaptivePerceptualCriterion):
    def __init__(self, dimension):
        super(ResidualAdaptivePerceptualCriterion, self).__init__()
        self.discriminator = ResidualDiscriminator(dimension)


class SqueezeExtractor(BasicMultiFeatureExtractor):
    def __init__(self, dimension, requires_grad=False):
        super(SqueezeExtractor, self).__init__(dimension, SQUEEZENET_CONFIG, requires_grad)


class SqueezeAdaptivePerceptualCriterion(ResidualAdaptivePerceptualCriterion):
    def __init__(self, dimension):
        super(SqueezeAdaptivePerceptualCriterion, self).__init__(dimension)
        self.features = SqueezeExtractor(dimension, requires_grad=True)
        self.features.to(self.device)
        self.predictor.to(self.device)


class SpectralDiscriminator(PerceptualDiscriminator):
    def __init__(self, dimension):
        super(SpectralDiscriminator, self).__init__(dimension)
        self.feat1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=dimension, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )

        self.feat2 = torch.nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.feat3 = torch.nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.feat4 = torch.nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.predictor = torch.nn.Sequential(
            SpectralNorm(nn.Conv2d(8, 1, 1, 1, 0, bias=False)),
            Flatten(),
        )


class SpectralAdaptivePerceptualCriterion(AdaptivePerceptualCriterion):
    def __init__(self, dimension):
        super(SpectralAdaptivePerceptualCriterion, self).__init__(dimension)
        self.bce = nn.BCEWithLogitsLoss()
        self.discriminator = SpectralDiscriminator(dimension)

    def update(self, actual, desire):
        self.discriminator.train()
        ploss, fake, real = self.featurize(actual, desire)
        self.LossD = self.feat(1.0 - real).mean() + self.feat(1.0 + fake).mean() + self.feat(self.margin - ploss).mean()
        return self.LossD

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        actual_features, _, ploss = self.featurize(actual, desire)
        self.lossG = ploss - self.predictor(actual_features[-1]).view(-1).mean() + self.distance(actual, desire)
        return self.lossG


class WassersteinAdaptivePerceptualCriterion(SpectralAdaptivePerceptualCriterion):
    def __init__(self, dimension):
        super(WassersteinAdaptivePerceptualCriterion, self).__init__(dimension)

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        ploss, result, _ = self.featurize(actual, desire)
        self.lossG = ploss - result.mean()
        return self.lossG

    def update(self, actual, desire):
        self.discriminator.train()
        ploss, fake, real = self.featurize(actual, desire)
        wgan_loss = fake.mean() - real.mean()
        alpha = float(random.uniform(0, 1))
        interpolates  = alpha * desire + (1 - alpha) * actual
        interpolates = Variable(interpolates.clone(), requires_grad=True).to(actual.device)
        _, interpolates_discriminator_out = self.discriminator(interpolates)
        buffer = Variable(torch.ones_like(interpolates_discriminator_out), requires_grad=True).to(actual.device)
        gradients = torch.autograd.grad(outputs=interpolates_discriminator_out, inputs=interpolates,grad_outputs=buffer, retain_graph=True, create_graph=True)[0]
        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        self.LossD = wgan_loss + 1e2*gradient_penalty
        return self.LossD
