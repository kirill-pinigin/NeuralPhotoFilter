import torch
import torch.nn as nn

from NeuralBlocks import HueSaturationValueCriterion

'''
https://en.wikipedia.org/wiki/Pyramid_(image_processing)
'''

class PyramidExtractor(nn.Module):
    def __init__(self):
        super(PyramidExtractor, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        g4 = self.downsample(x)
        u4 = torch.nn.functional.interpolate(g4, mode='bilinear', align_corners=True, scale_factor=2)
        g3 = self.downsample(g4)
        u3 = torch.nn.functional.interpolate(g3, mode='bilinear', align_corners=True, scale_factor=2)
        g2 = self.downsample(g3)
        u2 = torch.nn.functional.interpolate(g2, mode='bilinear', align_corners=True, scale_factor=2)
        g1 = self.downsample(g2)
        u1 = torch.nn.functional.interpolate(g1, mode='bilinear', align_corners=True, scale_factor=2)

        l1 = torch.abs(torch.sub(g2, u1))
        l2 = torch.abs(torch.sub(g3, u2))
        l3 = torch.abs(torch.sub(g4, u3))
        l4 = torch.abs(torch.sub(x,  u4))

        return l4, l3, l2, l1


class PyramidCriterion(nn.Module):
    def __init__(self):
        super(PyramidCriterion, self).__init__()
        self.features = PyramidExtractor()
        self.criterion = nn.MSELoss()

    def forward(self, actual, desire):
        actuals = self.features(actual)
        desires = self.features(desire)
        loss = 0.0

        for i in range(len(actuals)):
            loss += self.criterion(actuals[i], desires[i])

        del actuals, desires
        return loss


class ColorPyramidCriterion(PyramidCriterion):
    def __init__(self):
        super(ColorPyramidCriterion, self).__init__()
        self.HSV = HueSaturationValueCriterion()

    def forward(self, actual, desire):
        return super(ColorPyramidCriterion, self).forward(actual, desire) + self.HSV(actual, desire)
