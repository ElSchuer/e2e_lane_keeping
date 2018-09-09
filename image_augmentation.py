import cv2
import numpy as np
import matplotlib.pyplot as plt
import data_handler

def flip_horizontal(image, angle):
    return cv2.flip(image, flipCode=1), -angle

def manipulate_brightness(image, min_rand_val = 0, max_rand_val = 1):
    r = np.random.uniform(min_rand_val, max_rand_val)
    img = image.astype(np.float32)
    img[:,:,:] *= r
    np.clip(img, 0., 255.)
    return img.astype(np.uint8)

def random_shades(image):
    
    return image
