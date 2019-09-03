#!/usr/bin/env python3
import os
import cv2
import time
import pickle
import numpy as np
from scipy.linalg import norm
import Quartz.CoreGraphics as CG

wdir = os.path.dirname(__file__)
pickle_dir = os.path.join(os.path.join(os.path.dirname(wdir), 'data'), 'pickle')

with open(os.path.join(pickle_dir, 'digits.pkl'), 'rb') as pkl:
    DIGITS = pickle.load(pkl, encoding='latin1')

with open(os.path.join(pickle_dir, 'gameover.pkl'), 'rb') as pkl:
    GAMEOVER = pickle.load(pkl, encoding='latin1')

def compare(x, y):
    return norm(abs(x-y).ravel(), 0)

class Screenshot:

    def __init__(self, dy):
        self.dy = dy
        self.state = self._state()
        self.score = self._score()
        self.time = self._time()
        self.ring = self._rings()

    def window(self, x, y, width, height, dy):
        image = CG.CGWindowListCreateImage(CG.CGRectMake(x, y+dy, width, height),
                                           CG.kCGWindowListOptionOnScreenOnly,
                                           CG.kCGNullWindowID,
                                           CG.kCGWindowImageDefault)
        width = CG.CGImageGetWidth(image)
        height = CG.CGImageGetHeight(image)
        bytesperrow = CG.CGImageGetBytesPerRow(image)
        pixeldata = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image))
        X = np.frombuffer(pixeldata, dtype=np.uint8)
        X = X.reshape((height, bytesperrow//4, 4))
        return X[:, :, [2,1,0]][:, :width, :]

    def _state(self):
        return self.window(x=401, 
                           y=130, 
                           width=634, 
                           height=445, 
                           dy=0)

    def _score_array(self):
        X = self.window(x=434, 
                        y=142, 
                        width=190, 
                        height=88, 
                        dy=self.dy)
        r, g, b = X.T
        grey = (r == 196) & (g == 196) & (b == 196)
        X[:, :, :3][grey.T] = (0, 0, 0)
        other = (r > 0) & (g >= 0) & (b > 0)
        X[:, :, :3][other.T] = (255, 255, 255)
        return X

    def is_gameover(self):
        X = self.state
        X = X[828 + self.dy:852 + self.dy, 226:250, :3]
        return (compare(X, GAMEOVER) <= 1065)

    def digitizer(self, X):
        error = [compare(X, y) for y in DIGITS]
        if (min(error) < 300) and (min(error) != 432):
            return str(error.index(min(error)))
        else:
            return str('')

    def _score(self):
        X = self._score_array()[2:48, 0:379, :3]
        n1 = X[0+2:46, 348+4:379, :3]
        n2 = X[0+2:46, 316+4:347, :3]
        n3 = X[0+2:46, 284+4:315, :3]
        n4 = X[0+2:46, 252+4:283, :3]
        n5 = X[0+2:46, 220+4:251, :3]
        n6 = X[0+2:46, 188+4:219, :3]
        n7 = X[0+2:46, 156+4:187, :3]
        X = (n1, n2, n3, n4, n5, n6, n7)
        try:
            score = list(map(self.digitizer, X))[::-1]
            score = float(''.join(score).strip())
            return score
        except ValueError:
            return -1

    def _time(self):
        X = self._score_array()[66:112, 0:283, :3]
        n1 = X[0+2:46, 252+4:283, :3]
        n2 = X[0+2:46, 220+4:251, :3]
        n3 = X[0+2:46, 156+4:187, :3]
        X = (n1, n2, n3)
        time = list(map(self.digitizer, X))[::-1]
        if '' in time:
            return -1
        else:
            time = '{0}:{1}{2}'.format(time[0],time[1],time[2]).strip()
            return time

    def _rings(self):
        X = self._score_array()[130:176, 0:283, :3]
        n1 = X[0+2:46, 252+4:283, :3]
        n2 = X[0+2:46, 220+4:251, :3]
        n3 = X[0+2:46, 188+4:219, :3]
        X = (n1, n2, n3)
        try:
            rings = list(map(self.digitizer, X))[::-1]
            rings = float(''.join(rings).strip())
            return rings
        except ValueError:
            return -1


class ScreenshotProcessor:

    def __init__(self, resize_x, resize_y):
        self.resize_x = resize_x
        self.resize_y = resize_y

    def resize(self, X):
        return cv2.resize(X, (self.resize_x, self.resize_y))

    def grey(self, X):
        return cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
    
    def normalize(self, X):
        return X / 255

    def transform(self, X):
        X = self.resize(X)
        X = self.grey(X)
        X = self.normalize(X)
        return X