#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:39:30 2017

@author: dascienz
"""

import pickle
from scipy.linalg import norm

with open("/Users/dascienz/Desktop/sonicNN/pickles/digits.pkl","rb") as digits_pickle:
    digits = pickle.load(digits_pickle, encoding='latin1')

#score crop
def scoreRead(array,dy):
    #captures digit images on score line
    #digit farthest to the right is considered n1 and so on...    
    score = array[2+dy:48+dy, 0:379, :3]
    n1 = score[0+2:46, 348+4:379, :3]
    n2 = score[0+2:46, 316+4:347, :3]
    n3 = score[0+2:46, 284+4:315, :3]
    n4 = score[0+2:46, 252+4:283, :3]
    n5 = score[0+2:46, 220+4:251, :3]
    n6 = score[0+2:46, 188+4:219, :3]
    n7 = score[0+2:46, 156+4:187, :3]
    return n1,n2,n3,n4,n5,n6,n7
   
#timer crop
def timeRead(array,dy):
    #captures digit images on timer line
    #digit farthest to the right is considered n1 and so on...
    time = array[66+dy:112+dy, 0:283, :3]
    n1 = time[0+2:46, 252+4:283, :3]
    n2 = time[0+2:46, 220+4:251, :3]
    n3 = time[0+2:46, 156+4:187, :3]
    return n1,n2,n3

#rings crop 
def ringRead(array,dy):
    #captures digit images on rings line
    #digit farthest to the right is considered n1 and so on...
    rings = array[130+dy:176+dy, 0:283, :3]
    n1 = rings[0+2:46, 252+4:283, :3]
    n2 = rings[0+2:46, 220+4:251, :3]
    n3 = rings[0+2:46, 188+4:219, :3]
    return n1,n2,n3

def compare(a, b):
    diff = abs(a-b)
    return norm(diff.ravel(),0)

def digitRead(array):
    #recognize a digit from an image array 
    error = [compare(array, digits[i]) for i in range(10)]
    if (min(error) < 1000.0) and (min(error) != 432.0):
        return str(error.index(min(error)))
    else:
        return str("") 

def scoreBoard(array, dys, dyt, dyr): 
    s = list(map(digitRead, scoreRead(array,dys)))[::-1]
    t = list(map(digitRead, timeRead(array,dyt)))[::-1]
    r = list(map(digitRead, ringRead(array,dyr)))[::-1]
    try:
        score = float("".join(s).strip())
        time = (t[0]+":"+t[1]+t[2]).strip()
        rings = float("".join(r).strip())
        return (score,time,rings)
    except ValueError:
        return (0.0, 0.0, 0.0)

def gameOverRead(observation, dy):
    n = observation[828+dy:852+dy, 226:250, :3]
    return n
 
"""
IMPORTING FRAMES
"""
import glob
import numpy as np
from PIL import Image
path = "/Users/dascienz/Desktop/sonicNN/Test_Frames/*.png"
files = glob.glob(path)
frames = []
for frame in files:
    frames.append(Image.open(frame))
arrays = [np.array(frame) for frame in frames]

"""
IMAGING DIGITS
"""
import matplotlib.pyplot as plt
from PIL import Image

plt.xticks(np.arange(0,27,1))
plt.yticks(np.arange(0,44,1))
plt.grid(True)
plt.imshow(Image.fromarray(digits[7]))

plt.xticks(np.arange(0,27,1))
plt.yticks(np.arange(0,44,1))
plt.grid(True)
plt.imshow(Image.fromarray(scoreRead(arrays[8],0)[2]))


'''
PIXEL TRANSFORMATIONS IF STILL RGB VALUED
'''
"""
frames2 = frames 
arrays2 = [np.array(frame) for frame in frames2]
for array in arrays2:
    r, g, b = array.T
 
    grey = (r == 196) & (g == 196) & (b == 196)
    array[:,:,:3][grey.T] = (0,0,0)
    
    other = (r > 0) & (g >= 0) & (b > 0)
    array[:,:,:3][other.T] = (255,255,255)
  
for frame in frames2:
    pixels = frame.load()
    for x in range(frame.size[0]):
        for y in range(frame.size[1]):
            if pixels[x,y] == (196, 196, 196):
                pixels[x,y] = (0, 0, 0)
            else:
                pixels[x,y] = (255, 255, 255)
""" 
               
"""
test = [arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        arrays[4],
        arrays[5],
        arrays[6],
        arrays[7],
        arrays[8],
        arrays[9]]

digit_arrays = []
for i in xrange(10): 
    digit_arrays.append(timeRead(arrays[i],0)[0])

#write python arrays to file    
data = gameOverRead(arrays[1], 0)
output = open("/Users/dascienz/Desktop/sonicNN/gameover.pkl", "wb")
pickle.dump(data, output)
output.close()
"""