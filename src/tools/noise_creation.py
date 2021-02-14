import cv2
import numpy as np
from skimage.util import random_noise

##############################
# Noise functions
##############################

def noise_sp(img):  
    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(img, mode='s&p',amount=0.1)
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img

def noise_gaussian(img):
    # Generate Gaussian noise
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    # Add the Gaussian noise to the image
    noise_img = cv2.add(img,gauss)
    return noise_img


def noise_speckle(img):
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    noise_img = img + img * gauss
    return noise_img

def introduce_noise(img, noise_type="s&p"):
    functions_dict = {"s&p": noise_sp, "gaussian": noise_gaussian, "speckle": noise_speckle} 
    return functions_dict[noise_type](img)
