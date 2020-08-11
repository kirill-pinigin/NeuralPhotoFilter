import torch
import torch.nn as nn
import random

from NeuralBlocks import SpectralNorm, TotalVariation, HueSaturationValueCriterion
from PyramidCriterion import PyramidCriterion
from PerceptualCriterion import ChromaEdgePerceptualCriterion, FastNeuralStylePerceptualCriterion, EchelonPerceptualCriterion, MobilePerceptualCriterion, OxfordPerceptualCriterion, SharpPerceptualCriterion , SigmaPerceptualCriterion, SimplePerceptualCriterion, TubingenPerceptualCriterion
from SSIM import SSIMCriterion

'''
Super resolution with Generative Adversarial Networks
Boris Kovalenko
SUID 06201315
'''

class AdversarialCriterion(nn.Module):
    def __init__(self, dimension,  weight : float = 1e-2):
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
            nn.Sigmoid(),
        )

        self.perceptualizer = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, actual, desire):
        return  self.evaluate(actual, desire), self.update(actual, desire)

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        result = self.discriminator(actual).view(-1)
        ones = torch.ones(result.shape).to(actual.device)
        return self.perceptualizer(actual, desire) + self.weight * self.bce(result, ones)

    def update(self, actual, desire):
        self.discriminator.train()
        real = self.discriminator(desire.detach()).view(-1)
        ones = torch.ones(real.shape).to(actual.device)
        fake = self.discriminator(actual.detach()).view(-1)
        zeros = torch.zeros(fake.shape).to(actual.device)
        return self.bce(real, ones) + self.bce(fake, zeros)


class AdversarialStyleCriterion(AdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(AdversarialStyleCriterion, self).__init__(dimension, weight)
        self.perceptualizer = FastNeuralStylePerceptualCriterion( dimension, weight)
        self.distance = nn.L1Loss()

    def evaluate(self, actual, desire):
        return super(AdversarialStyleCriterion, self).evaluate(actual, desire) + self.distance(actual, desire)


'''
DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks
Andrey Ignatov, Nikolay Kobyshev, Kenneth Vanhoey, Radu Timofte, Luc Van Gool
ETH Zurich
'''

class DSLRAdversaialCriterion(AdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(DSLRAdversaialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = SharpPerceptualCriterion(dimension)
        self.pyramid = PyramidCriterion()
        self.tv = TotalVariation()

    def evaluate(self, actual, desire):
        return  super(DSLRAdversaialCriterion, self).evaluate(actual, desire) \
                    + self.pyramid(actual, desire) \
                    + self.tv(actual)


class MobileImprovingAdversarialCriterion(AdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(MobileImprovingAdversarialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = MobilePerceptualCriterion(dimension)
        self.ssim = SSIMCriterion(dimension)
        self.tv = TotalVariation()
        self.distance = nn.L1Loss()

    def evaluate(self, actual, desire):
        return super(MobileImprovingAdversarialCriterion, self).evaluate(actual, desire) \
                     + self.distance(actual, desire) + self.ssim(actual, desire) \
                     + self.tv(actual)


class MultiSigmaCriterion(MobileImprovingAdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(MultiSigmaCriterion, self).__init__(dimension, weight)
        self.perceptualizer = SigmaPerceptualCriterion(dimension, weight)


class EchelonAdversaialCriterion(MobileImprovingAdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(EchelonAdversaialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = EchelonPerceptualCriterion(dimension, weight)


class TubingenAdversaialCriterion(MobileImprovingAdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(TubingenAdversaialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = TubingenPerceptualCriterion(dimension)


class ChromaAdversarialCriterion(MobileImprovingAdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(ChromaAdversarialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = ChromaEdgePerceptualCriterion(dimension, weight)
        self.HSV = HueSaturationValueCriterion()

    def evaluate(self, actual, desire):
        return super(ChromaAdversarialCriterion, self).evaluate(actual, desire) \
                     + self.HSV(actual, desire)


class OxfordAdversarialCriterion(AdversarialStyleCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(OxfordAdversarialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = OxfordPerceptualCriterion( dimension)
        self.distance = nn.MSELoss()


'''
Image-to-Image Translation with Conditional Adversarial Networks
Phillip Isola Jun-Yan Zhu Tinghui Zhou Alexei A. Efros
Berkeley AI Research (BAIR) Laboratory, UC Berkeley
'''

class PatchAdversarialCriterion(AdversarialCriterion):
    def __init__(self, dimension):
        super(PatchAdversarialCriterion, self).__init__(dimension)
        self.perceptualizer = nn.L1Loss()

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        rest =self.discriminator(actual).view(-1)
        ones = torch.ones(rest.shape).to(actual.device)
        return 1e2*self.perceptualizer(actual, desire) + self.bce(rest, ones)


class PatchColorAdversarialCriterion(PatchAdversarialCriterion):
    def __init__(self, dimension,):
        super(PatchColorAdversarialCriterion, self).__init__(dimension)
        self.perceptualizer = HueSaturationValueCriterion()
        self.chroma_edge = ChromaEdgePerceptualCriterion(dimension)

    def evaluate(self, actual, desire):
        return super(PatchColorAdversarialCriterion, self).evaluate(actual, desire) \
                     + self.chroma_edge(actual, desire)

'''
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network
Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, ´
Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi
Twitter
'''

class PhotoRealisticAdversarialCriterion(AdversarialCriterion):
    def __init__(self, dimension, weight : float = 1e-2):
        super(PhotoRealisticAdversarialCriterion, self).__init__(dimension, weight)
        self.perceptualizer = SimplePerceptualCriterion(dimension)

'''
SPECTRAL NORMALIZATION
FOR GENERATIVE ADVERSARIAL NETWORKS
Takeru Miyato1
, Toshiki Kataoka1
, Masanori Koyama2
, Yuichi Yoshida3
{miyato, kataoka}@preferred.jp
koyama.masanori@gmail.com
yyoshida@nii.ac.jp
1Preferred Networks, Inc. 2Ritsumeikan University 3National Institute of Informatics
'''

class SpectralAdversarialCriterion(AdversarialCriterion):
    def __init__(self, dimension):
        super(SpectralAdversarialCriterion, self).__init__(dimension)
        self.perceptualizer = SimplePerceptualCriterion(dimension)
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
        )

        self.relu = nn.ReLU()

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        return 1e2*self.perceptualizer(actual, desire) - self.discriminator(actual).mean()

    def update(self, actual, desire):
        self.discriminator.train()
        real = self.discriminator(desire.detach())
        fake = self.discriminator(actual.detach())
        return self.relu(1.0 - real).mean() + self.relu(1.0 + fake).mean()


'''
Improved Training of Wasserstein GANs
Ishaan Gulrajani1∗
, Faruk Ahmed1
, Martin Arjovsky2
, Vincent Dumoulin1
, Aaron Courville1
 Montreal Institute for Learning Algorithms
2Courant Institute of Mathematical Sciences
'''

class WassersteinAdversarialCriterion(PatchAdversarialCriterion):
    def __init__(self, dimension):
        super(WassersteinAdversarialCriterion, self).__init__(dimension)
        self.perceptualizer = SharpPerceptualCriterion(dimension)

    def evaluate(self, actual, desire):
        self.discriminator.eval()
        return self.perceptualizer(actual, desire) - self.discriminator(actual).mean()

    def update(self, actual, desire):
        self.discriminator.train()
        real = self.discriminator(desire.detach()).view(-1)
        fake = self.discriminator(actual.detach()).view(-1)
        alpha = float(random.uniform(0, 1))
        interpolates = alpha * desire.detach().clone() + (1 - alpha) * actual.detach().clone()
        interpolates = torch.autograd.Variable(interpolates.to(actual.device), requires_grad=True)
        x  = self.discriminator(interpolates).view(-1)
        buffer = torch.autograd.Variable(torch.ones_like(x).to(actual.device), requires_grad=True)
        gradients = torch.autograd.grad(outputs=x, inputs=interpolates, grad_outputs=buffer, retain_graph=True, create_graph=True)[0]
        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        return fake.mean() - real.mean() + 1e2 * gradient_penalty
