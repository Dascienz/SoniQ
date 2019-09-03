#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 18:03:45 2018

@author: Dascienz
"""

import cv2
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def img_import():
    path = "/Users/dascienz/Desktop/sonic_screenshots/*.png"
    files = glob.glob(path)
    frames = []
    for frame in files:
        frames.append(Image.open(frame).convert('RGB'))
    arrays = [np.array(frame) for frame in frames]
    return frames, arrays

def show_image(array):
    """Show screenshot."""
    plt.imshow(Image.fromarray(array))
    return
    
def show_resize(array):
    """Resize screenshot."""
    array = cv2.resize(array, (80,80))
    plt.imshow(Image.fromarray(array))
    return array

def show_resize_grayscale(array):
    """Uncolor screenshot."""
    array = cv2.cvtColor(show_resize(array), cv2.COLOR_RGB2GRAY)
    plt.imshow(Image.fromarray(array))
    return array  


def preprocess(observation):
    """Pre-process image array to (80*80) grayscale image array."""
    observation = cv2.resize(observation, (80, 80))
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    _, observation = cv2.threshold(observation, 80, 255, cv2.THRESH_BINARY)
    observation = cv2.Canny(observation, 1, 255)
    return observation


if __name__ == "__main__":
    
    frames, arrays = img_import()
    
    #show_image(arrays[1]); plt.show();
    
    #show_resize(arrays[1]); plt.show();
    
    #show_resize_grayscale(arrays[1]); plt.show();
    
    #array = show_resize_grayscale(arrays[1])/255.0
    plt.imshow(Image.fromarray(preprocess(arrays[4])))
