import random
from PIL import Image
import numbers
import torchvision.transforms.functional as F


class DualComposeTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input_image, target_image):
        for t in self.transforms:
            input_image, target_image = t(input_image, target_image)
        return  input_image, target_image

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class DualToTensor(object):
    def __call__(self, input_image, target_image):
        return F.to_tensor(input_image), F.to_tensor(target_image)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class DualRandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, input_image, target_image):
        if self.padding is not None:
            input_image = F.pad(input_image, self.padding, self.fill, self.padding_mode)
            target_image = F.pad(target_image, self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        if self.pad_if_needed and input_image.size[0] < self.size[1]:
            input_image = F.pad(input_image, (self.size[1] - input_image.size[0], 0), self.fill, self.padding_mode)
            target_image = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and input_image.size[1] < self.size[0]:
            input_image = F.pad(input_image, (0, self.size[0] - input_image.size[1]), self.fill, self.padding_mode)
            target_image = F.pad(target_image, (0, self.size[0] - target_image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(input_image, self.size)
        return F.crop(input_image, i, j, h, w), F.crop(target_image, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class DualCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, input, target):
        return F.center_crop(input, self.size), F.center_crop(target, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class DualResize(object):
    def __init__(self, size : int, interpolation=Image.BICUBIC):
        self.size = (size, size)
        self.interpolation = interpolation

    def __call__(self, input_image, target_image):
        return F.resize(input_image, self.size, self.interpolation), F.resize(target_image, self.size, self.interpolation)


class DualRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, input_image, target_image):
        if random.random() < self.p:
            return F.hflip(input_image), F.hflip(target_image)
        return input_image, target_image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class DualRandomVerticalFlip(object):
    def __init__(self,  p=0.5):
        self.p = p

    def __call__(self, input_image, target_image):
        if random.random() < self.p:
            return F.vflip(input_image), F.hflip(target_image)
        return input_image, target_image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class DualRandomRotation(object):
    def __init__(self, degrees, resample=Image.BICUBIC, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, input_image, target_image):
        angle = self.get_params(self.degrees)
        return F.rotate(input_image, angle, self.resample, self.expand, self.center),  F.rotate(target_image, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
