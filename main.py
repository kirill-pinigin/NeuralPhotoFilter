import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn

from AdelaideGenerator import AdelaideGenerator, AdelaideFastGenerator, AdelaideResidualGenerator, AdelaideSupremeGenerator
from AdversarialCriterion import AdversarialCriterion, SubSampleAdversaialCriterion, EchelonAdversaialCriterion, AdversarialStyleCriterion, ChromaAdversarialCriterion,  DSLRAdversaialCriterion, PatchAdversarialCriterion, PatchColorAdversarialCriterion, MultiSigmaCriterion, MobileImprovingAdversarialCriterion, PhotoRealisticAdversarialCriterion, SpectralAdversarialCriterion, WassersteinAdversarialCriterion
from BerkeleyGenerator import BerkeleyGenerator, BerkeleyFastGenerator, BerkeleyResidualGenerator, BerkeleySupremeGenerator
from NeuralPhotoFilter import NeuralPhotoFilter
from Dataset import  DeblurDataset, DenoiseDataset, Image2ImageDataset, ColorizationDataset, UpscalingDataset
from FreiburgGenerator import FreiburgGenerator, FreiburgResidualGenerator, FreiburgSupremeGenerator, FreiburgSqueezeGenerator
from MovaviGenerator import MovaviGenerator, MovaviFastGenerator,  MovaviResidualGenerator, MovaviStrongGenerator, MovaviSupremeGenerator
from NeuralBlocks import SILU, UpsampleDeConv, TransposedDeConv, PixelDeConv
from AdaptivePerceptualCriterion import AdaptivePerceptualCriterion, SqueezeAdaptivePerceptualCriterion, SpectralAdaptivePerceptualCriterion, WassersteinAdaptivePerceptualCriterion
from SSIM import SSIM
from StanfordGenerator import   StanfordGenerator, StanfordFastGenerator,  StanfordModernGenerator,  StanfordStrongGenerator, StanfordSupremeGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--dimension',         type = int,   default=1, help='must be equal 1 for grayscale or 3 for RGB')
parser.add_argument('--image_size',        type = int,   default=256, help='pixel size of square image')
parser.add_argument('--image_dir',         type = str,   default='./AustoRestorerEntireDataset300/', help='path to dataset')
parser.add_argument('--operation',         type = str,   default='Restoration', help='type of deconvolution')
parser.add_argument('--generator',         type = str,   default='MovaviSupreme', help='type of image generator')
parser.add_argument('--criterion',         type = str,   default='MobileImproving', help='type of criterion')
parser.add_argument('--deconv',            type = str,   default='Upsample', help='type of deconv')
parser.add_argument('--activation',        type = str,   default='Leaky', help='type of activation')
parser.add_argument('--optimizer',         type = str,   default='Adam', help='type of optimizer')
parser.add_argument('--batch_size',        type = int,   default=256)
parser.add_argument('--epochs',            type = int,   default=256)
parser.add_argument('--lr',                type = float, default=1e-3)
parser.add_argument('--resume_train',      type = bool,  default=True)

args = parser.parse_args()
assert args.dimension == 1 or args.dimension == 3

print(torch.__version__)

criterion_types =   {
                        'Adversarial'           : AdversarialCriterion,
                        'AdversarialStyle'      : AdversarialStyleCriterion,
                        'Chroma'                : ChromaAdversarialCriterion,
                        'DSLR'                  : DSLRAdversaialCriterion,
                        'Echelon'               : EchelonAdversaialCriterion,
                        'MultiSigma'            : MultiSigmaCriterion,
                        'MobileImproving'       : MobileImprovingAdversarialCriterion,
                        'Patch'                 : PatchAdversarialCriterion,
                        'PatchColor'            : PatchColorAdversarialCriterion,
                        'PhotoRealistic'        : PhotoRealisticAdversarialCriterion,
                        'Spectral'              : SpectralAdversarialCriterion,
                        'SubSample'             : SubSampleAdversaialCriterion,
                        'Wasserstein'           : WassersteinAdversarialCriterion,
                        'PAN'                   : AdaptivePerceptualCriterion,
                        'SpectralPAN'           : SpectralAdaptivePerceptualCriterion,
                        'SqueezeAdaptive'       : SqueezeAdaptivePerceptualCriterion,
                        'WassersteinAdaptive'   : WassersteinAdaptivePerceptualCriterion,
                    }

generator_types = {
                        'Adelaide'          : AdelaideGenerator,
                        'AdelaideFast'      : AdelaideFastGenerator,
                        'AdelaideResidual'  : AdelaideResidualGenerator,
                        'AdelaideSupreme'   : AdelaideSupremeGenerator,
                        'Berkeley'          : BerkeleyGenerator,
                        'BerkeleyFast'      : BerkeleyFastGenerator,
                        'BerkeleyResidual'  : BerkeleyResidualGenerator,
                        'BerkeleySupreme'   : BerkeleySupremeGenerator,
                        'Freiburg'          : FreiburgGenerator,
                        'FreiburgResidual'  : FreiburgResidualGenerator,
                        'FreiburgSqueeze'   : FreiburgSqueezeGenerator,
                        'FreiburgSupreme'   : FreiburgSupremeGenerator,
                        'Stanford'          : StanfordGenerator,
                        'StanfordFast'      : StanfordFastGenerator,
                        'StanfordModern'    : StanfordModernGenerator,
                        'StanfordStrong'    : StanfordStrongGenerator,
                        'StanfordSupreme'   : StanfordSupremeGenerator,
                        'Movavi'            : MovaviGenerator,
                        'MovaviFast'        : MovaviFastGenerator,
                        'MovaviResidual'    : MovaviResidualGenerator,
                        'MovaviStrong'      : MovaviStrongGenerator,
                        'MovaviSupreme'     : MovaviSupremeGenerator,
                    }

operation_types =   {
                        'Coloriztion'       : ColorizationDataset,
                        'Deblur'            : DeblurDataset,
                        'Denoise'           : DenoiseDataset,
                        'Restoration'       : Image2ImageDataset,
                        'Upscaling'         : UpscalingDataset,
                    }

deconv_types =      {
                        'Transposed'  : TransposedDeConv,
                        'Upsample'    : UpsampleDeConv,
                        'Pixel'       : PixelDeConv
                    }

activation_types =  {
                        'ReLU' : nn.ReLU(),
                        'Leaky': nn.LeakyReLU(),
                        'PReLU': nn.PReLU(),
                        'ELU'  : nn.ELU(),
                        'SELU' : nn.SELU(),
                        'SILU' : SILU()
                    }

optimizer_types =   {
                        'Adam'   : optim.Adam,
                        'RMSprop': optim.RMSprop,
                        'SGD'    : optim.SGD
                    }
accuracy = SSIM(args.dimension)
model = generator_types[args.generator]
deconvLayer = (deconv_types[args.deconv] if args.deconv in deconv_types else deconv_types['upsample'])
function = (activation_types[args.activation] if args.activation in activation_types else activation_types['Leaky'])
generator = model(dimension=args.dimension, deconv=deconvLayer, activation=function)
optimizer =(optimizer_types[args.optimizer] if args.optimizer in optimizer_types else optimizer_types['Adam'])(generator.parameters(), lr = args.lr)
criterion = criterion_types[args.criterion](dimension=args.dimension)
deconvolution_dataset = operation_types[args.operation]

augmentations = {'train' : True, 'val' : False}
shufles = {'train' : True, 'val' : False}
batch_sizes = {'train' : args.batch_size, 'val' : args.batch_size}

image_datasets = {x: deconvolution_dataset(args.dimension, args.image_size, os.path.join(args.image_dir, x),  augmentation = augmentations[x])
                    for x in ['train', 'val']}

imageloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                             shuffle=shufles[x], num_workers=4)
                for x in ['train', 'val']}

test_dataset = deconvolution_dataset(args.dimension, args.image_size, args.image_dir+'/val/',  augmentation = False)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

framework = NeuralPhotoFilter(generator = generator, criterion = criterion, accuracy=accuracy, optimizer = optimizer)
framework.approximate(dataloaders = imageloaders, num_epochs=args.epochs, resume_train=args.resume_train)
framework.estimate(testloader)
