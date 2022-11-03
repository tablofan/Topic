import cv2
import numpy as np
import os

im = []
folder = "Raphael"
for filepath in os.listdir(folder):
    im.append(cv2.imread(f'{folder}/{filepath}', 0))


