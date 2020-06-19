import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

from NeuralBlocks import  SpectralNorm, TotalVariation, HueSaturationValueCriterion, Flatten
from PyramidCriterion import PyramidCriterion
from PerceptualCriterion import  ChromaEdgePerceptualCriterion, FastNeuralStylePerceptualCriterion , EchelonPerceptualCriterion, MobilePerceptualCriterion, SubSamplePerceptualCriterion, SharpPerceptualCriterion , SigmaPerceptualCriterion, SimplePerceptualCriterion
from SSIM import SSIMCriterion


class AdversarialCriterion(nn.Module):
    def __init__(self, dimension,  weight : float = 1e-3):
        super(AdversarialCriterion, self).__init__()
        self.weight = weight
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=dimension, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False),
            Flatten(),
            nn.Sigmoid(),
        )

        self.perceptualizer = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.lossG = None
        self.lossD = None

    def forward(self, actual, desire):
        return  self.evaluate(actual, desire), self.update(actual, desire)

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        result = self.discriminator(actual)
        ones = Variable(torch.ones(result.shape).to(actual.device))
        self.lossG = self.perceptualizer(actual, desire) + self.weight * self.bce(result, ones)
        return self.lossG

    def update(self, actual, desire):
        self.discriminator.train()
        real = self.discriminator(desire.detach())
        ones = Variable(torch.ones(real.shape).to(actual.device))
        fake =  self.discriminator(actual.detach())
        zeros = Variable(torch.zeros(fake.shape).to(actual.device))
        lossADreal = self.bce(real, ones)
        lossADfake = self.bce(fake, zeros)
        self.lossD = lossADreal + lossADfake
        return self.lossD

    def backward(self, retain_variables=True):
        return self.lossG.backward(retain_variables=retain_variables)


class AdversarialStyleCriterion(AdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(AdversarialStyleCriterion, self).__init__(dimension, weight)
        self.perceptualizer = FastNeuralStylePerceptualCriterion( dimension, weight)
        self.distance = nn.L1Loss()

    def evaluate(self, actual, desire):
        self.lossG = super(AdversarialStyleCriterion, self).forward(actual, desire) + self.distance(actual, desire)
        return self.lossG


class DSLRAdversaialCriterion(AdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(DSLRAdversaialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = SharpPerceptualCriterion(dimension, weight)
        self.discriminator.to(self.device)
        self.pyramid = PyramidCriterion()
        self.tv = TotalVariation()

    def evaluate(self, actual, desire):
        self.lossG = super(DSLRAdversaialCriterion, self).forward(actual, desire) \
                    + self.pyramid(actual, desire) \
                    + self.tv(actual)
        return self.lossG


class MobileImprovingAdversarialCriterion(AdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(MobileImprovingAdversarialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = MobilePerceptualCriterion(dimension)
        self.ssim = SSIMCriterion(dimension)
        self.tv = TotalVariation()
        self.distance = nn.L1Loss()

    def evaluate(self, actual, desire):
        self.lossG = super(MobileImprovingAdversarialCriterion, self).forward(actual, desire) + self.distance(actual, desire) + self.ssim(actual, desire)
        return self.lossG


class MultiSigmaCriterion(MobileImprovingAdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(MultiSigmaCriterion, self).__init__(dimension, weight)
        self.perceptualizer = SigmaPerceptualCriterion(dimension, weight)


class EchelonAdversaialCriterion(MobileImprovingAdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(EchelonAdversaialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = EchelonPerceptualCriterion(dimension, weight)


class SubSampleAdversaialCriterion(MobileImprovingAdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(SubSampleAdversaialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = SubSamplePerceptualCriterion(dimension, weight)


class ChromaAdversarialCriterion(MobileImprovingAdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(ChromaAdversarialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = ChromaEdgePerceptualCriterion(dimension, weight)
        self.HSV = HueSaturationValueCriterion()

    def evaluate(self, actual, desire):
        self.lossG = super(ChromaAdversarialCriterion, self).forward(actual, desire) + self.HSV(actual, desire)
        return self.lossG


class PatchAdversarialCriterion(AdversarialCriterion):
    def __init__(self,dimension):
        super(PatchAdversarialCriterion, self).__init__(dimension)
        self.perceptualizer = EchelonPerceptualCriterion(dimension)

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        rest =self.discriminator(actual)
        ones = Variable(torch.ones(rest.shape).to(actual.device))
        self.lossG = 1e2*self.perceptualizer(actual, desire) + self.bce(rest, ones)
        return self.lossG


class PatchColorAdversarialCriterion(PatchAdversarialCriterion):
    def __init__(self, dimension,):
        super(PatchColorAdversarialCriterion, self).__init__(dimension)
        self.perceptualizer = HueSaturationValueCriterion()
        self.chroma_edge = ChromaEdgePerceptualCriterion()

    def forward(self, actual, desire):
        self.lossG = super(PatchColorAdversarialCriterion, self).forward(actual, desire) + self.chroma_edge(actual, desire)
        return self.lossG


class PhotoRealisticAdversarialCriterion(AdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(PhotoRealisticAdversarialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = SimplePerceptualCriterion()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=dimension, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=1, bias=False),
            Flatten(),
        )
        self.discriminator.to(self.device)

    def evaluate(self, actual, desire):
        self.lossG = super(PhotoRealisticAdversarialCriterion, self).forward(actual, desire) + nn.functional.mse_lossA(actual, desire)
        return self.lossG


class SpectralAdversarialCriterion(nn.Module):
    def __init__(self, dimension):
        super(SpectralAdversarialCriterion, self).__init__()
        self.perceptualizer = MobilePerceptualCriterion(dimension)
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=dimension, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNorm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNorm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            SpectralNorm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False),
            Flatten(),
        )

        self.relu = nn.ReLU()
        self.lossG = None

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        self.lossG = 1e2*self.perceptualizer(actual, desire) - self.discriminator(actual).view(-1).mean()
        return self.lossG

    def update(self, actual, desire):
        self.discriminator.train()
        real = self.discriminator(desire.detach())
        fake = self.discriminator(actual.detach())
        lossDreal = self.relu(1.0 - real).mean()
        lossDfake = self.relu(1.0 + fake).mean()
        self.lossD = lossDreal + lossDfake
        return self.lossD

    def backward(self, retain_variables=True):
        return self.lossG.backward(retain_variables=retain_variables)


class WassersteinAdversarialCriterion(AdversarialCriterion):
    def __init__(self, dimension):
        super(WassersteinAdversarialCriterion, self).__init__(dimension)
        self.perceptualizer = FastNeuralStylePerceptualCriterion(dimension, 1e-2)
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=dimension, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=False),
            Flatten(),
        )

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        self.lossG = 1e2*self.perceptualizer(actual, desire) - self.discriminator(actual).view(-1).mean()
        return self.lossG

    def update(self, actual, desire):
        self.discriminator.train()
        real = self.discriminator(desire.detach()).view(-1)
        fake = self.discriminator(actual.detach()).view(-1)
        wgan_lossA = fake.mean() - real.mean()
        alpha = float(random.uniform(0,1))
        interpolates  = alpha * desire + (1 - alpha) * actual
        interpolates = Variable(interpolates.clone(), requires_grad=True).to(actual.device)
        interpolates_discriminator_out  = self.discriminator(interpolates).view(-1)

        buffer = Variable(torch.ones_like(interpolates_discriminator_out), requires_grad=True).to(actual.device)
        gradients = torch.autograd.grad(outputs=interpolates_discriminator_out, inputs=interpolates,
                                  grad_outputs=buffer,
                                  retain_graph=True,
                                  create_graph=True)[0]

        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        self.lossD =  wgan_lossA + 1e2*gradient_penalty
        return self.lossD
