import torch
import torch.nn as nn
from torchvision import models
import random

from NeuralBlocks import  SpectralNorm

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
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
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
        self.factors = [1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0]

    def forward(self, actual, desire):
        return self.evaluate(actual, desire), self.update(actual, desire)

    def featurize(self, actual, desire):
        actual_features, a = self.discriminator(actual)
        desire_features, d = self.discriminator(desire)
        ploss = 0.0

        for i in range(len(desire_features)):
            ploss += self.factors[i]*self.distance(actual_features[i], desire_features[i])

        del actual_features, desire_features
        return ploss, a, d

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        ploss, rest, _ = self.featurize(actual, desire)
        ones = torch.ones(rest.shape).to(actual.device)
        return ploss + self.bce(rest, ones) + self.distance(actual, desire)

    def update(self, actual, desire):
        self.discriminator.train()
        ploss, fake, real = self.featurize(actual.detach(), desire.detach())
        zeros = torch.zeros(fake.shape).to(actual.device)
        ones = torch.ones(real.shape).to(actual.device)
        return self.bce(real, ones) + self.bce(fake, zeros) + self.relu(self.margin - ploss).mean()


class OxfordDiscriminator(PerceptualDiscriminator):
    def __init__(self, dimension):
        super(OxfordDiscriminator, self).__init__(dimension)
        features = models.vgg11_bn(pretrained=True).features
        conv = nn.Conv2d(dimension, 64, kernel_size=3, padding=1)

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

        self.feat1 = torch.nn.Sequential()
        self.feat2 = torch.nn.Sequential()
        self.feat3 = torch.nn.Sequential()
        self.feat4 = torch.nn.Sequential()

        self.feat1.add_module(str(0), conv)

        for x in range(1, 4):
            self.feat1.add_module(str(x), features[x])

        for x in range(4, 8):
            self.feat2.add_module(str(x), features[x])

        for x in range(8, 15):
            self.feat3.add_module(str(x), features[x])

        for x in range(15, 22):
            self.feat4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = True


class OxfordAdaptivePerceptualCriterion(AdaptivePerceptualCriterion):
    def __init__(self, dimension):
        super(OxfordAdaptivePerceptualCriterion, self).__init__(dimension)
        self.discriminator = OxfordDiscriminator(dimension)


class ResidualDiscriminator(PerceptualDiscriminator):
    def __init__(self, dimension):
        super(ResidualDiscriminator, self).__init__(dimension)
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

        self.feat1 = nn.Sequential(
            conv,
            base_model.bn1,
            nn.ReLU(),
            base_model.maxpool,
        )

        self.feat2 = base_model.layer1
        self.feat3 = base_model.layer2
        self.feat4 = base_model.layer3

        self.predictor = torch.nn.Sequential(
            base_model.layer4,
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )

        for param in self.parameters():
            param.requires_grad = True


class ResidualAdaptivePerceptualCriterion(AdaptivePerceptualCriterion):
    def __init__(self, dimension):
        super(ResidualAdaptivePerceptualCriterion, self).__init__(dimension)
        self.discriminator = ResidualDiscriminator(dimension)


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
            SpectralNorm(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.predictor = torch.nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)),
        )


class SpectralAdaptivePerceptualCriterion(AdaptivePerceptualCriterion):
    def __init__(self, dimension):
        super(SpectralAdaptivePerceptualCriterion, self).__init__(dimension)
        self.bce = nn.BCEWithLogitsLoss()
        self.discriminator = SpectralDiscriminator(dimension)

    def update(self, actual, desire):
        self.discriminator.train()
        ploss, fake, real = self.featurize(actual.detach(), desire.detach())
        return self.relu(1.0 - real).mean() + self.relu(1.0 + fake).mean() + self.relu(self.margin - ploss).mean()

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        ploss, result, _ = self.featurize(actual, desire)
        return ploss - result.view(-1).mean() + self.distance(actual, desire)


class WassersteinAdaptivePerceptualCriterion(SpectralAdaptivePerceptualCriterion):
    def __init__(self, dimension):
        super(WassersteinAdaptivePerceptualCriterion, self).__init__(dimension)

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        ploss, result, _ = self.featurize(actual, desire)
        return ploss - result.mean()

    def update(self, actual, desire):
        self.discriminator.train()
        ploss, fake, real = self.featurize(actual.detach(), desire.detach())
        alpha = float(random.uniform(0, 1))
        interpolates = alpha * desire.detach().clone() + (1 - alpha) * actual.detach().clone()
        interpolates = torch.autograd.Variable(interpolates.to(actual.device), requires_grad=True)
        _, interpolates_discriminator_out = self.discriminator(interpolates)
        buffer = torch.ones_like(interpolates_discriminator_out).to(actual.device)
        gradients = torch.autograd.grad(outputs=interpolates_discriminator_out, inputs=interpolates,grad_outputs=buffer, retain_graph=True, create_graph=True)[0]
        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        return fake.mean() - real.mean() + 1e2 * gradient_penalty
