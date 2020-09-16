from torchvision import transforms
import os
from PIL import Image
import numpy as np
import cv2
from scipy import signal
from math import ceil
import random

#psfs_path = './psf_kernels.dat'
class RandomMotionBlur(object):
    def __init__(self, image_size, dimension=3, trajectory_iters=2000, trajectory_canvas=64, psf_canvas=64,
                 psfs_path = './psf_kernels.dat', max_len=60, count_psfs_file=1250):
        self.list_random = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
        self.image_size = image_size
        self.psf_size = psf_canvas
        self.trajectory_size = trajectory_canvas
        self.trajectory_it = trajectory_iters
        self.trajectory_maxlen = max_len
        self.directory_psfs = psfs_path
        self.count_kernels = count_psfs_file
        self.dimension = dimension
        self.result = None

    def __call__(self, photo):
        return self.blur_image(photo)

    def blur_psf(self, psf, delta, array_photo):
        tmp = np.pad(psf, delta // 2, 'constant')
        cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        blured = cv2.normalize(array_photo, array_photo, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
        if self.dimension != 1:
            blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
            blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))

        blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        blured = np.asarray(np.abs(blured) * 255, dtype=np.uint8)
        return Image.fromarray(blured)

    def blur_image(self, photo):
        if self.directory_psfs is None:
            psf = random.choice(RandomMotionBlur.generate_PSF(canvas=self.psf_size,
                                                              trajectory=RandomMotionBlur.generate_trajectory(
                                                                  canvas=self.trajectory_size, iters=self.trajectory_it,
                                                                  max_len=self.trajectory_maxlen,
                                                                  expl=np.random.choice(self.list_random))))
        else:
            psf = self.random_choice_psf_fromfile()

        original = photo.copy()

        if self.dimension == 1:
            original = np.expand_dims(original, axis=-1)

        key, kex = psf.shape
        delta = self.image_size - key
        assert delta >= 0, 'Разрешение изображения должно быть больше ядра'

        original = np.array(original, dtype=np.float32)
        blur = self.blur_psf(psf=psf, delta=delta, array_photo=original)
        self.result = blur
        return self.result

    @staticmethod
    def generate_trajectory(canvas=64, iters=2000, max_len=60, expl=None):
        expl = 0.1 * np.random.uniform(0, 1) if expl is None else expl
        tot_length, big_expl_count = 0, 0
        centripetal = 0.7 * np.random.uniform(0, 1)
        prob_big_shake = 0.2 * np.random.uniform(0, 1)
        gaussian_shake = 10 * np.random.uniform(0, 1)
        init_angle = 360 * np.random.uniform(0, 1)

        img_v0 = np.sin(np.deg2rad(init_angle))
        real_v0 = np.cos(np.deg2rad(init_angle))

        v0 = complex(real=real_v0, imag=img_v0)
        v = v0 * max_len / (iters - 1)

        if expl > 0:
            v = v0 * expl

        x = np.array([complex(real=0, imag=0)] * iters)

        for t in range(0, iters - 1):
            if np.random.uniform() < prob_big_shake * expl:
                next_direction = 2 * v * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
                big_expl_count += 1
            else:
                next_direction = 0

            dv = next_direction + expl * (gaussian_shake * complex(
                real=np.random.randn(), imag=np.random.randn()) - centripetal * x[t]) * (max_len / (iters - 1))

            v += dv
            v = (v / float(np.abs(v))) * (max_len / float((iters - 1)))
            x[t + 1] = x[t] + v
            tot_length = tot_length + abs(x[t + 1] - x[t])

        x += complex(real=-np.min(x.real), imag=-np.min(x.imag))
        x = x - complex(real=x[0].real % 1., imag=x[0].imag % 1.) + complex(1, 1)
        x += complex(real=ceil((canvas - max(x.real)) / 2), imag=ceil((canvas - max(x.imag)) / 2))
        return iters, x

    def generate_sequence_psf(self):
        psf_list = []
        for i in range(self.count_kernels):
            psf = random.choice(RandomMotionBlur.generate_PSF(canvas=self.psf_size,
                                                              trajectory=RandomMotionBlur.generate_trajectory(
                                                                  canvas=self.trajectory_size, iters=self.trajectory_it,
                                                                  max_len=self.trajectory_maxlen,
                                                                  expl=np.random.choice(self.list_random))))
            psf_list.append(psf.astype(np.float32).flatten())
        psf_list = np.array(psf_list)
        psf_list.tofile(self.directory_psfs)

    def random_choice_psf_fromfile(self):
        with open(self.directory_psfs, 'rb') as fp:
            fp.seek(np.random.randint(low=0, high=self.count_kernels) * 4 * 64 * 64, os.SEEK_SET)
            kernel = np.fromfile(fp, dtype=np.float32, count=64 * 64)
        kernel = kernel.reshape((self.psf_size, self.psf_size))
        return kernel

    def __repr__(self):
        return str(self.__dict__)

    @staticmethod
    def generate_PSF(canvas=None, trajectory=None, fraction=None):
        canvas = (canvas, canvas)
        iters, traj_x = RandomMotionBlur.generate_trajectory(canvas=canvas, expl=0.005) if trajectory is None \
            else trajectory
        fraction = [1 / 100, 1 / 10, 1 / 2, 1] if fraction is None else fraction

        PSFnumber = len(fraction)
        PSFs = []

        PSF = np.zeros(canvas)

        triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
        triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))

        for j in range(PSFnumber):
            prevT = 0 if j == 0 else fraction[j - 1]

            for t in range(iters):
                if (fraction[j] * iters >= t) and (prevT * iters < t - 1):
                    t_proportion = 1
                elif (fraction[j] * iters >= t - 1) and (prevT * iters < t - 1):
                    t_proportion = fraction[j] * iters - (t - 1)
                elif (fraction[j] * iters >= t) and (prevT * iters < t):
                    t_proportion = t - (prevT * iters)
                elif (fraction[j] * iters >= t - 1) and (prevT * iters < t):
                    t_proportion = (fraction[j] - prevT) * iters
                else:
                    t_proportion = 0

                m2 = int(np.minimum(canvas[1] - 1, np.maximum(1, np.math.floor(traj_x[t].real))))
                M2 = int(m2 + 1)
                m1 = int(np.minimum(canvas[0] - 1, np.maximum(1, np.math.floor(traj_x[t].imag))))
                M1 = int(m1 + 1)

                PSF[m1, m2] += t_proportion * triangle_fun_prod(traj_x[t].real - m2, traj_x[t].imag - m1)
                PSF[m1, M2] += t_proportion * triangle_fun_prod(traj_x[t].real - M2, traj_x[t].imag - m1)
                PSF[M1, m2] += t_proportion * triangle_fun_prod(traj_x[t].real - m2, traj_x[t].imag - M1)
                PSF[M1, M2] += t_proportion * triangle_fun_prod(traj_x[t].real - M2, traj_x[t].imag - M1)

            PSFs.append(PSF / iters)

        return np.array(PSFs)
