import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn

from AdaptivePerceptualCriterion import AdaptivePerceptualCriterion, OxfordAdaptivePerceptualCriterion , ResidualAdaptivePerceptualCriterion, SpectralAdaptivePerceptualCriterion, WassersteinAdaptivePerceptualCriterion
from AdelaideGenerator import AdelaideGenerator, AdelaideFastGenerator, AdelaideResidualGenerator, AdelaideSupremeGenerator
from AdversarialCriterion import AdversarialCriterion, EchelonAdversaialCriterion, AdversarialStyleCriterion, ChromaAdversarialCriterion,  DSLRAdversaialCriterion, PatchAdversarialCriterion, PatchColorAdversarialCriterion, DeblurSimpleCriterion, MultiSigmaCriterion, MobileImprovingAdversarialCriterion, OxfordAdversarialCriterion, PhotoRealisticAdversarialCriterion, TubingenAdversaialCriterion, SpectralAdversarialCriterion, WassersteinAdversarialCriterion, DeblurWassersteinAdversarialCriterion
from BerkeleyGenerator import BerkeleyGenerator, BerkeleyFastGenerator, BerkeleyResidualGenerator, BerkeleySupremeGenerator
from Dataset import  DeblurDataset, DenoiseDataset, Image2ImageDataset, ColorizationDataset, UpscalingDataset
from FreiburgGenerator import FreiburgGenerator, FreiburgFastGenerator, FreiburgAttentiveGenerator, FreiburgModernGenerator, FreiburgResidualGenerator, FreiburgSupremeGenerator, FreiburgSqueezeGenerator
from MovaviGenerator import MovaviGenerator, MovaviFastGenerator,  MovaviResidualGenerator, MovaviStrongGenerator, MovaviSupremeGenerator
from NeuralBlocks import SILU, UpsampleDeConv, TransposedDeConv, PixelDeConv
from NeuralPhotoFilter import NeuralPhotoFilter
from SSIM import SSIM
from StanfordGenerator import   StanfordGenerator, StanfordFastGenerator,  StanfordModernGenerator, StanfordStrongGenerator, StanfordSupremeGenerator
from TexasGenerator import TexasResidualGenerator, TexasSupremeGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--operation',         type = str,   default='MotionDeblur', help='type of deconvolution')
parser.add_argument('--image_dir',         type = str,   default='./MotionDeblurMixedRealDataset300/', help='path to dataset')
parser.add_argument('--dimension',         type = int,   default=3, help='must be equal 1 for grayscale or 3 for RGB')
parser.add_argument('--image_size',        type = int,   default=256, help='pixel size of square image')
parser.add_argument('--generator',         type = str,   default='TexasSupreme', help='type of image generator')
parser.add_argument('--criterion',         type = str,   default='DeblurWasserstein', help='type of criterion')
parser.add_argument('--deconv',            type = str,   default='Upsample', help='type of deconv')
parser.add_argument('--activation',        type = str,   default='ReLU', help='type of activation')
parser.add_argument('--optimizer',         type = str,   default='Adam', help='type of optimizer')
parser.add_argument('--batch_size',        type = int,   default=128)
parser.add_argument('--epochs',            type = int,   default=256)
parser.add_argument('--resume_train',      type = bool,  default=True)

args = parser.parse_args()
assert args.dimension == 1 or args.dimension == 3

print(torch.__version__)

operation_types =   {
                        'Colorization'      : ColorizationDataset,
                        'Deblur'            : DeblurDataset,
                        'Denoise'           : DenoiseDataset,
                        'MotionDeblur'      : Image2ImageDataset,
                        'Restoration'       : Image2ImageDataset,
                        'Upscaling'         : UpscalingDataset,
                    }

criterion_types =   {
                        'Adversarial'           : AdversarialCriterion,
                        'AdversarialStyle'      : AdversarialStyleCriterion,
                        'Chroma'                : ChromaAdversarialCriterion,
                        'DeblurSimple'          : DeblurSimpleCriterion,
                        'DeblurWasserstein'     : DeblurWassersteinAdversarialCriterion,
                        'DSLR'                  : DSLRAdversaialCriterion,
                        'Echelon'               : EchelonAdversaialCriterion,
                        'MultiSigma'            : MultiSigmaCriterion,
                        'MobileImproving'       : MobileImprovingAdversarialCriterion,
                        'Oxford'                : OxfordAdversarialCriterion,
                        'OxfordAdaptive'        : OxfordAdaptivePerceptualCriterion,
                        'PAN'                   : AdaptivePerceptualCriterion,
                        'Patch'                 : PatchAdversarialCriterion,
                        'PatchColor'            : PatchColorAdversarialCriterion,
                        'PhotoRealistic'        : PhotoRealisticAdversarialCriterion,
                        'ResidualAdaptive'      : ResidualAdaptivePerceptualCriterion,
                        'Spectral'              : SpectralAdversarialCriterion,
                        'SpectralPAN'           : SpectralAdaptivePerceptualCriterion,
                        'Tubingen'              : TubingenAdversaialCriterion,
                        'Wasserstein'           : WassersteinAdversarialCriterion,
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
                        'FreiburgFast'      : FreiburgFastGenerator,
                        'FreiburgResidual'  : FreiburgResidualGenerator,
                        'FreiburgSqueeze'   : FreiburgSqueezeGenerator,
                        'FreiburgSupreme'   : FreiburgSupremeGenerator,
                        'Stanford'          : StanfordGenerator,
                        'StanfordFast'      : StanfordFastGenerator,
                        'FreiburgAttentive' : FreiburgAttentiveGenerator,
                        'FreiburgModern'    : FreiburgModernGenerator,
                        'StanfordModern'    : StanfordModernGenerator,
                        'StanfordStrong'    : StanfordStrongGenerator,
                        'StanfordSupreme'   : StanfordSupremeGenerator,
                        'Movavi'            : MovaviGenerator,
                        'MovaviFast'        : MovaviFastGenerator,
                        'MovaviResidual'    : MovaviResidualGenerator,
                        'MovaviStrong'      : MovaviStrongGenerator,
                        'MovaviSupreme'     : MovaviSupremeGenerator,
                        'TexasResidual'     : TexasResidualGenerator,
                        'TexasSupreme'      : TexasSupremeGenerator,
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
criterion = criterion_types[args.criterion](dimension=args.dimension)
deconvolution_dataset = operation_types[args.operation]

augmentations = {'train' : True, 'val' : False}
shufles = {'train' : True, 'val' : False}

image_datasets = {x: deconvolution_dataset(args.dimension, args.image_size, os.path.join(args.image_dir, x), augmentation = augmentations[x])
                    for x in ['train', 'val']}

image_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=shufles[x], num_workers=torch.cuda.device_count())
                for x in ['train', 'val']}

framework = NeuralPhotoFilter(generator = generator, criterion = criterion, accuracy=accuracy, dimension=args.dimension, image_size=args.image_size)
framework.approximate(dataloaders = image_loaders, num_epochs=args.epochs, resume_train=args.resume_train)
framework.estimate(image_loaders['val'])
