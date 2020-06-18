from math import log10, exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable

def PSNR(actual, desire):
    mse = F.mse_loss(actual, desire)
    return 10 * log10(1 / mse.item())


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, full= False):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    #ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


class SSIM(torch.nn.Module):
    def __init__(self, dimension):
        super(SSIM, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv2d = torch.nn.Conv2d(dimension, dimension, kernel_size=11, padding=11//2, groups=dimension, bias= False)
        self.conv2d.weight.data=create_window(11, dimension)
        self.conv2d.to(self.device)
        self.C1 = float(0.01 ** 2)
        self.C2 = float(0.03 ** 2)

    def forward(self, img1, img2):
        mu1 =  self.conv2d(img1)
        mu2 =  self.conv2d(img2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.conv2d(img1 * img1) - mu1_sq
        sigma2_sq = self.conv2d(img2 * img2) - mu2_sq
        sigma12 = self.conv2d(img1 * img2) - mu1_mu2

        # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        cs_map = (2 * sigma12 + self.C2) / (sigma1_sq + sigma2_sq + self.C2)  # set alpha=beta=gamma=1
        ssim_map = ((2 * mu1_mu2 + self.C1) / (mu1_sq + mu2_sq + self.C1)) * cs_map

        return ssim_map.mean()


class SSIMCriterion(SSIM):
    def __init__(self, dimension):
        super(SSIMCriterion, self).__init__(dimension)

    def forward(self, actual, desire):
        return 1.0 - super(SSIMCriterion, self).forward(actual, desire)


class MultiScaleSSIMCriterion(torch.nn.Module):
    def __init__(self):
        super(MultiScaleSSIMCriterion, self).__init__()
        self.criterion = SSIM()
        self.factors = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    def forward(self, actual, desire):
        mssim = []
        mcs = []
        img1 = actual.clone()
        img2 = desire.clone()

        for _ in range(len(self.factors)):
            sim, cs = self.criterion (img1, img2)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)

        mssim = (mssim + 1.0) / 2.0
        mcs = (mcs + 1.0) / 2.0

        pow1 = mcs ** self.factors
        pow2 = mssim ** self.factors
        output = torch.prod(pow1[:-1] * pow2[-1])
        return 1.0 - output

