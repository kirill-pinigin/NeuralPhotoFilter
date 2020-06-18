import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch.nn.parameter import Parameter
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

        feats = {
            'feat1': feat1,
            'feat2': feat2,
            'feat3': feat3,
            'feat4': feat4,
        }
        return feats, result

class AdaptivePerceptualCriterion(nn.Module):
    def __init__(self, dimension):
        super(AdaptivePerceptualCriterion, self).__init__()
        self.discriminator = PerceptualDiscriminator(dimension)
        self.distance = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.feat = nn.ReLU()
        self.margin = 1.0
        self.lossP = None
        self.lossD = None

    def evaluate(self, actual, desire):
        actual_features, a = self.discriminator(actual)
        desire_features, d = self.discriminator(desire)
        ploss = 0.0

        for i in range(len(desire_features)):
            ploss += self.factors[i]*self.ContentCriterion(actual_features[i], desire_features[i])

        return ploss, a, d
'''
    def meta_optimize(self, lossD, length):
        self.current_loss += float(lossD.item()) / length

        if self.counter > ITERATION_LIMIT:
            self.current_loss = self.current_loss / float(ITERATION_LIMIT)
            if self.current_loss < self.best_loss:
                self.best_loss = self.current_loss
                print('! best_loss !', self.best_loss)
            else:
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                    if lr >= LR_THRESHOLD:
                        param_group['lr'] = lr * 0.2
                        print('! Decrease LearningRate in Perceptual !', lr)
            self.counter = int(0)
            self.current_loss = float(0)

        self.counter += int(1)
'''
    def update(self, actual, desire):
        self.discriminator.train()
        ploss, fake, real = self.evaluate(actual, desire)
        zeros = Variable(torch.zeros(fake.shape).to(actual.device))
        ones = Variable(torch.ones(real.shape).to(actual.device))
        lossDreal = self.AdversarialCriterion(real, ones)
        lossDfake = self.AdversarialCriterion(fake, zeros)
        self.LossD = lossDreal + lossDfake + self.feat(self.margin - ploss).mean()
        return self.LossD

    def forward(self, actual, desire):
        self.discriminator.eval()
        ploss, rest, _ = self.evaluate(actual, desire)
        ones = Variable(torch.ones(rest.shape).to(actual.device))
        aloss = self.AdversarialCriterion(rest, ones)
        self.LossP = ploss + aloss + self.ContentCriterion(actual, desire)
        return self.LossP

    def backward(self, retain_variables=True):
        return self.LossD.backward(retain_variables=retain_variables)


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
            Flatten
        )


class SpectralAdaptivePerceptualCriterion(AdaptivePerceptualCriterion):
    def __init__(self):
        super(SpectralAdaptivePerceptualCriterion, self).__init__()
        self.AdversarialCriterion = nn.BCEWithLogitsCriterion()
        self.discriminator = SpectralDiscriminator()

    def update(self, actual, desire):
        self.discriminator.train()
        ploss, fake, real = self.evaluate(actual, desire)
        lossDreal = self.feat(1.0 - real).mean()
        lossDfake = self.feat(1.0 + fake).mean()

        self.LossD = lossDreal + lossDfake + self.feat(self.margin - ploss).mean()
        return self.LossD

    def forward(self, actual, desire):
        self.predictor.eval()
        self.features.eval()
        actual_features, _, ploss = self.evaluate(actual, desire)
        self.LossP = ploss - self.predictor(actual_features[-1]).view(-1).mean() + self.ContentCriterion(actual, desire)
        return self.LossP


class WassersteinAdaptivePerceptualCriterion(SpectralAdaptivePerceptualCriterion):
    def __init__(self):
        super(WassersteinAdaptivePerceptualCriterion, self).__init__()

    def forward(self, actual, desire):
        self.discriminator.eval()
        ploss, result, _ = self.evaluate(actual, desire)
        self.LossP = ploss - result.mean()
        return self.LossP

    def update(self, actual, desire):
        self.discriminator.train()
        ploss, fake, real = self.evaluate(actual, desire)

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