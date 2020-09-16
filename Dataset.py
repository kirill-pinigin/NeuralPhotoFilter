import torch.utils.data as data
import torchvision
from torchvision import transforms
import os
import random
import numpy as np
from os import listdir
from os.path import join
import cv2
from PIL import ImageFilter, ImageEnhance, Image
#from PyBlur import RandomizedBlur
from RandomMotionBlur import RandomMotionBlur

from DualTransform import DualComposeTransforms,  DualToTensor, DualRandomCrop, DualRandomHorizontalFlip, DualResize

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_image(filepath, dimension, image_size, augmentation : bool = True):
    if dimension == 3:
        image = Image.open(filepath).convert('RGB')
    else :
        image = Image.open(filepath).convert('YCbCr')
        image, _, _ = image.split()

    if augmentation:
        image = image.resize((int(image_size * 1.171875) , int(image_size * 1.171875)), Image.BICUBIC)

    return image


class Image2ImageDataset(data.Dataset):
    def __init__(self, dimension, image_size, image_dir, augmentation: bool = False):
        super(Image2ImageDataset, self).__init__()
        self.dimension = dimension
        self.image_size = image_size
        self.augmentation = augmentation
        self.deprocess = False

        transforms_list = [
            DualResize(image_size),
            DualToTensor(),
        ]

        if augmentation:
            transforms_list = [
                                  DualRandomCrop(image_size),
                                  DualRandomHorizontalFlip(),
                              ] + transforms_list

        self.transforms =  DualComposeTransforms(transforms_list)
        dirs = get_immediate_subdirectories(image_dir)
        dirs.sort()
        self.inputs  = [join(image_dir + '/' + dirs[0], x) for x in listdir(image_dir + '/' + dirs[0]) if is_image_file(x)]
        self.targets = [join(image_dir + '/' + dirs[1], x) for x in listdir(image_dir + '/' + dirs[1]) if is_image_file(x)]
        print(len(self.inputs))
        print(len(self.targets))

    def __getitem__(self, index):
        input = load_image(self.inputs[index], self.dimension,  self.image_size, self.augmentation)
        target = load_image(self.targets[index], self.dimension,  self.image_size, self.augmentation)
        input, target = self.transforms(input, target)
        return input, target

    def __len__(self):
        return len(self.inputs)


class DistortDataset(data.Dataset):
    def __init__(self, dimension, image_size, image_dir, augmentation: bool = False):
        super(DistortDataset, self).__init__()
        self.dimension = dimension
        self.image_size = image_size
        self.distorter = transforms.Compose([RandomSharp()])
        self.deprocess = False
        self.augmentation = None
        self.images = []

        if augmentation:
            self.augmentation = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2),
            ])

        self.tensoration = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        for subdir, dirs, files in os.walk(image_dir):
            for filename in files:
                filepath = subdir + os.sep + filename
                if filepath.endswith(".jpg") or filepath.endswith(".png"):
                    self.images.append(filepath)

    def __getitem__(self, index):
        target = load_image(self.images[index],  self.dimension, self.image_size, self.augmentation)

        if self.augmentation is not None:
            target = self.augmentation(target)

        input = self.distorter(target)
        input, target = self.tensoration(input), self.tensoration(target)
        return input, target

    def __len__(self):
        return len(self.images)


class DeblurDataset(DistortDataset):
    def __init__(self, dimension, image_size, image_dir, augmentation: bool = False):
        super(DeblurDataset, self).__init__(dimension, image_size, image_dir, augmentation)
        self.distorter = transforms.Compose([ RandomMotionBlur(image_size=image_size, dimension=dimension)])

    def __getitem__(self, index):
        target = load_image(self.images[index],  self.dimension, self.image_size, self.augmentation)

        if self.augmentation is not None:
            target = self.augmentation(target)

        input = self.distorter(target)
        input, target = self.tensoration(input), self.tensoration(target)
        return input, target


class DenoiseDataset(DistortDataset):
    def __init__(self, dimension, image_size, image_dir, augmentation: bool = False):
        super(DenoiseDataset, self).__init__(dimension, image_size, image_dir, augmentation)
        self.distorter = transforms.Compose([RandomNoise()])
        self.preparation = transforms.Compose([transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),  transforms.ToTensor()])

    def denoise(self, image):
        image = image.convert('RGB')
        image = np.array(image)
        out = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        result = Image.fromarray(out)

        if self.dimension == 1:
            result = result.convert('L')

        return result

    def __getitem__(self, index):
        target = load_image(self.images[index], self.dimension, self.image_size, self.augmentaion)
        
        if self.augmentation is not None:
            target = self.augmentation(target)
        
        distorted = self.distorter(target)
        input, target = self.tensoration(distorted), self.tensoration(target)
        
        if self.deprocess:
            denoised = self.denoise(distorted)
            return input, target, self.preparation(denoised)
        else:
            return input, target


class ColorizationDataset(DistortDataset):
    def __init__(self, dimension, image_size, image_dir, augmentation):
        super(ColorizationDataset, self).__init__(dimension, image_size, image_dir, augmentation)
        self.distorter = transforms.Compose([RandomSharp()])

    def __getitem__(self, index):
        target = load_image(self.images[index], self.dimension, self.image_size, self.augmentation)

        if  self.augmentation is not None:
            target = self.augmentation(target)

        target = np.array(target)
        target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(target)
        input = Image.fromarray(cv2.cvtColor(l, cv2.COLOR_GRAY2RGB))
        input = self.distorter(input)
        target = cv2.cvtColor(target, cv2.COLOR_LAB2RGB)
        input, target = self.tensoration(input), self.tensoration(Image.fromarray(target))
        return input, target


class UpscalingDataset(DeblurDataset):
    def __init__(self, dimension, image_size, image_dir, augmentation):
        super(UpscalingDataset, self).__init__(dimension, image_size, image_dir, augmentation)
        self.distorter = transforms.Compose([RandomResize(image_size)])


class RandomNoise(object):
    def __init__(self):
        self.sigmas = [14, 15]
        self.random_state= np.random.RandomState(42)

    def __call__(self, img):
        transforms = []
        noise = np.random.choice([1, 2 , 3])
        dispersion = random.uniform(self.sigmas [0], self.sigmas[1])

        if noise == 1:
            transforms.append(Lambda(lambda img: self.camera_noise(img, dispersion)))

        elif noise == 2:
            transforms.append(Lambda(lambda img: self.perlin_noise(img, dispersion)))
            transforms.append(Lambda(lambda img: self.luma_noise(img, dispersion)))
            random.shuffle(transforms)

        else:
            transforms.append(Lambda(lambda img: self.gaussian_noise(img, dispersion)))
            random.shuffle(transforms)

        transform = torchvision.transforms.Compose(transforms)
        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += ', sigma={0}'.format(self.sigma)
        return format_string

    def camera_noise(self, input, sigma):
        if sigma > 0:
            img = np.array(input)
            img = img.astype(dtype=np.float32)
            photons = self.random_state.poisson(img, size=img.shape)
            electrons = 0.69 * photons
            additive_noise = self.random_state.normal(scale=sigma, size=electrons.shape)
            additive_noise = cv2.GaussianBlur(additive_noise,(0,0), 0.3)
            noisy_img =  electrons + additive_noise
            noisy_img  = (noisy_img * 1.33).astype(np.int)
            noisy_img += 6
            noisy_img = np.clip(noisy_img, 0, 255)
            noisy_img = noisy_img.astype(dtype=np.uint8)
            return Image.fromarray(noisy_img)
        else:
            return input

    def perlin_noise(self, input, sigma):
        if sigma > 0:
            input = np.array(input)
            yuv = cv2.cvtColor(input, cv2.COLOR_RGB2YUV)
            img = yuv[:, :, 0]
            img = img.astype(dtype=np.float32)
            noise = generate_perlin_noise(input , 2)
            noisy_img =  img + noise.astype(dtype=np.float32)*sigma
            noisy_img = np.clip(noisy_img, 0.0, 255.0)
            noisy_img = noisy_img.astype(dtype=np.uint8)
            yuv[:, :, 0] = noisy_img
            return Image.fromarray(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB))
        else:
            return input

    def luma_noise(self, input, sigma):
        if sigma > 0:
            input = np.array(input)
            yuv = cv2.cvtColor(input, cv2.COLOR_RGB2YUV)
            img = yuv[:, :, 0]
            img = img.astype(dtype=np.float32)
            photons = self.random_state.poisson(img, size=img.shape)
            electrons = 0.69 * photons
            additive_noise = self.random_state.normal(scale=sigma, size=electrons.shape)
            additive_noise = cv2.GaussianBlur(additive_noise,(0,0), 0.3)
            noisy_img =  electrons + additive_noise
            noisy_img  = (noisy_img * 1.33).astype(np.int)
            noisy_img += 6
            noisy_img = np.clip(noisy_img, 0, 255)
            noisy_img = noisy_img.astype(dtype=np.uint8)
            yuv[:, :, 0] = noisy_img
            return Image.fromarray(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB))
        else:
            return input

    def chroma_noise(self, input, sigma):
        if sigma > 0:
            img = np.array(input)
            yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            u = yuv[:, :, 1]
            v = yuv[:, :, 2]
            u = u.astype(dtype=np.float32)
            v = v.astype(dtype=np.float32)
            n_u = np.random.normal(0.0, sigma, u.shape)
            n_v = np.random.normal(0.0, sigma, v.shape)
            n_u = cv2.blur(n_u,(5,5))
            n_v = cv2.blur(n_v,(5,5))
            u =  u + n_u
            v =  v + n_v
            u = np.clip(u, 0.0, 255.0)
            v = np.clip(v, 0.0, 255.0)
            u = u.astype(dtype=np.uint8)
            v = v.astype(dtype=np.uint8)
            yuv[:, :, 1] = u
            yuv[:, :, 2] = v
            return Image.fromarray(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB))
        else:
            return input

    def gaussian_noise(self, input, sigma):
        if sigma > 0:
            input = np.array(input)
            img = input.astype(dtype=np.float32)
            noisy_img = img + np.random.normal(0.0, sigma, img.shape)
            noisy_img = np.clip(noisy_img, 0.0, 255.0)
            noisy_img = noisy_img.astype(dtype=np.uint8)
            return Image.fromarray(noisy_img)
        else:
            return input

    def poisson_noise(self, input, sigma):
        image = np.array(input)
        vals = len(np.unique(image))
        vals = float(2 ** np.ceil(np.log2(vals)))
        image = image.astype(dtype=np.float32)
        image = np.random.poisson(image * vals) / vals
        noisy_img = image + np.random.normal(0.0, sigma, image.shape)
        noisy_img = np.clip(noisy_img, 0.0, 255.0)
        noisy_img = noisy_img.astype(dtype=np.uint8)
        return Image.fromarray(noisy_img)


        if sigma > 0:
            input = np.array(input)
            yuv = cv2.cvtColor(input, cv2.COLOR_RGB2YUV)
            image = yuv[:, :, 0]
            vals = len(np.unique(image))
            vals = float(2 ** np.ceil(np.log2(vals)))
            image = image.astype(dtype=np.float32)

            image = np.random.poisson(image * vals) / vals
            noisy_img = image + np.random.normal(0.0, sigma, image.shape)
            noisy_img = np.clip(noisy_img, 0.0, 255.0)
            noisy_img = noisy_img.astype(dtype=np.uint8)
            yuv[:, :, 0] = noisy_img
            return Image.fromarray(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB))
        else:
            return input

    def uniform_noise(self, input, sigma):
        if sigma > 0:
            input = np.array(input)
            img = input.astype(dtype=np.float32)
            noisy_img = img + np.random.uniform(-sigma, sigma, img.shape)
            noisy_img = np.clip(noisy_img, 0.0, 255.0)
            noisy_img = noisy_img.astype(dtype=np.uint8)
            return Image.fromarray(noisy_img)
        else:
            return input

    def speckle_noise(self, input, sigma):
        if sigma > 0:
            input = np.array(input)
            yuv = cv2.cvtColor(input, cv2.COLOR_RGB2YUV)
            img = yuv[:, :, 0]
            img = img.astype(dtype=np.float32)
            noisy_img = img + np.random.normal(0.0, sigma, img.shape)*img  / 127.5
            noisy_img = np.clip(noisy_img, 0.0, 255.0)
            noisy_img = noisy_img.astype(dtype=np.uint8)
            yuv[:, :, 0] = noisy_img
            return Image.fromarray(cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB))
        else:
            return input


class RandomResize(object):
    def __init__(self, image_size):
        random.seed(42)
        self.image_size = image_size

    def __call__(self, input):
        factor = random.randint(2, 12)
        factor *= 0.5
        image = torchvision.transforms.functional.resize(input, (int(self.image_size / factor), int(self.image_size / factor)), Image.BICUBIC)
        result = torchvision.transforms.functional.resize(image, (int(self.image_size), int(self.image_size)), Image.BICUBIC)
        return result

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.factors)


class RandomSharp(object):
    def __init__(self, factor = 0.5):
        self.range = [0.0, factor]

    def __call__(self, input):
        enhancer = ImageEnhance.Sharpness(input)
        img = enhancer.enhance(random.uniform( self.range[0],  self.range[1]))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.factors)

from itertools import product, count

def generate_unit_vectors(n, sigma= 2*np.pi):
    'Generates matrix NxN of unit length vectors'
    phi = np.random.uniform(0, sigma, (n, n))
    v = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    return v

def qz(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def cz(t):
    return -2 * t * t * t + 3 * t * t


def generate_perlin_noise(size, ns, sigma = 2*np.pi):
    nc = int(size / ns)
    grid_size = int(size / ns + 1)
    v = generate_unit_vectors(grid_size, sigma)
    ad, ar = np.arange(ns), np.arange(-ns, 0, 1)
    vd = np.zeros((ns, ns, 4, 1, 2))
    for (l1, l2), c in zip(product((ad, ar), repeat=2), count()):
        vd[:, :, c, 0] = np.stack(np.meshgrid(l2, l1, indexing='xy'), axis=2)

    d = qz(np.stack((np.zeros((ns, ns, 2)),
                     np.stack(np.meshgrid(ad, ad, indexing='ij'), axis=2)),
                    axis=2) / ns)
    d[:, :, 0] = 1 - d[:, :, 1]
    d0 = d[..., 0].copy().reshape(ns, ns, 1, 2)
    d1 = d[..., 1].copy().reshape(ns, ns, 2, 1)
    m = np.zeros((size, size))
    t = m.reshape(nc, ns, nc, ns)

    for i, j in product(np.arange(nc), repeat=2):
        av = v[i:i+2, j:j+2].reshape(4, 2, 1)
        at = np.matmul(vd, av).reshape(ns, ns, 2, 2)
        t[i, :, j, :] = np.matmul(np.matmul(d0, at), d1).reshape(ns, ns)
    return m


class Lambda(object):
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
