import numpy as np
from .DefocusBlur import  DefocusBlur_random
from .LinearMotionBlur import LinearMotionBlur_random
from .PsfBlur import PsfBlur_random

blurFunctions = {"0": DefocusBlur_random, "1": LinearMotionBlur_random, "2": PsfBlur_random}

def RandomizedBlur(img):
    blurToApply = blurFunctions[str(np.random.randint(0, len(blurFunctions)))]
    return blurToApply(img)
