import os

import cv2
import mahotas
import numpy as np
import pandas as pd
from skimage.util import random_noise
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tools.noise_creation import *

####################
# Feature descriptors
####################

# Greycolor Histogram
def histogram(img, mask=None, bins=8):
    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Compute the color histogram
    hist  = cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # Normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()

# Color histogram, return the 3 color histograms concatenated
def color_histogram(img, mask=None, bins=8):
    histograms={'b':np.zeros(255), 'g':np.zeros(255), 'r':np.zeros(255)}
    for j,col in enumerate(histograms.keys()):
        # Compute the color histogram for each color
        hist  = cv2.calcHist([img], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        # Normalize each color
        histograms[col]=cv2.normalize(hist, hist).flatten()
    return np.hstack([histograms["r"],histograms["g"],histograms["b"]])

# Hu Moments
def hu_moments(image):
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute HuMoments
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Haralick Texture
def haralick(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute haralick texture
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

#power spectrum slope, contrast, noise and saturation 
# Power spectrum slope
def power_spectrum(img):
    # Resize the image to have a shorter power spectrum feature
    img = cv2.resize(img, tuple((50, 50)))
    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Calculate Fourier Transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    #Flatten the feature vector
    flatted_spectrum = magnitude_spectrum.flatten()
    #Normalize the feature vector
    flatted_spectrum = flatted_spectrum/np.linalg.norm(flatted_spectrum)
    return flatted_spectrum


##############################
# Feature extraction per path
##############################

def global_feature_extraction(path_split, set_type, option = 0,noise=False):
    all_names=[]
    path_set = os.path.join(path_split,set_type)
    labels_type = os.listdir(path_set)
    fixed_size = tuple((500, 500))
    labels=[]
    features=[]
    # loop over the training data sub-folders
    for label in labels_type:
        
        # Get paths
        label_dir = os.path.join(path_set, label)
        image_names_label = os.listdir(label_dir)
        all_names.append(image_names_label)
        
        # Loop over each image in each label
        for image_name in image_names_label:
            # Get image path
            image_path = os.path.join(label_dir, image_name)

            # Read the image and resize it to a fixed-size
            image = cv2.imread(image_path)
            image = cv2.resize(image, fixed_size)

            # Introduce noise if indicated
            if noise:
                image = introduce_noise(image, noise)

            # Features extraction and concatenation
            hu_moments_vector = hu_moments(image)    # Shape
            if option == 0:
                histogram_vector  = histogram(image)     #Color
                haralick_vector   = haralick(image)      # Texture
                global_feature = np.hstack([histogram_vector, hu_moments_vector, haralick_vector])
            if option == 1:
                power_spectrum_vector = power_spectrum(image)   #FFT
                color_histogram_vector = color_histogram(image) #Color
                global_feature = np.hstack([color_histogram_vector, power_spectrum_vector, np.repeat(hu_moments_vector,100)])
           
            # Update lists of labels and feature vectors
            labels.append(label)
            features.append(global_feature)
    print(f'[INFO] Finished {set_type} feature extraction')

    # Encode labels
    le          = LabelEncoder()
    labels      = le.fit_transform(labels)

    # Normalize feature vector
    scaler            = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(features)

    return labels, rescaled_features, [item for sublist in all_names for item in sublist]
