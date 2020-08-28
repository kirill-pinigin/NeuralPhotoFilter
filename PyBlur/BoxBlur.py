import numpy as np
from PIL import Image
import cv2

boxKernelDims = [3,5,7,9]

def BoxBlur_random(img):
    kernelidx = np.random.randint(0, len(boxKernelDims))    
    kerneldim = boxKernelDims[kernelidx]
    return BoxBlur(img, kerneldim)

def BoxBlur(img, dim):
    imgarray = np.array(img, dtype="float32")
    kernel = BoxKernel(dim)
    convolved = cv2.filter2D(imgarray, -1, kernel).astype("uint8")
    img = Image.fromarray(convolved)
    return img

def BoxKernel(dim):
    kernelwidth = dim
    kernel = np.ones((kernelwidth, kernelwidth), dtype=np.float32)        
    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel