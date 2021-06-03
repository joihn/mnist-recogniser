from labelLoader import Loader
import torch
from lenet5_like import LeNet5_like
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

import os
import skimage.filters as fl
from skimage import exposure as ex
from skimage.transform import rescale, resize
from skimage.morphology import disk
import numpy as np
import pickle
import cv2
# import imutils
import cv2 as cv
from numpy.linalg import eig
from skimage.transform import rotate
from skimage import measure, img_as_float
from scipy.stats import multivariate_normal
import matplotlib.patches as mpatches
import matplotlib.colors as cl
from skimage import filters as fl
from skimage.filters import threshold_otsu
import matplotlib.patches as pat
import pandas as pd
import torch

def getSuitsData():
    loader = Loader("C:/Users/maxim/Google Drive/Epfl/MA4/Img analysis/project/iapr/project/train_games/")

    #%%

    nP = 4
    nR = 13
    nG = 7
    # trainlabels = torch.empty((nG, nR, nP))
    # trainImg = torch.empty((nG, nR, nP, 1, 28, 28))

    inter_img = []
    for g in range(1, nG + 1):
        for r in range(1, nR + 1):
            for p in range(1, nP + 1):
                if loader.getSuit(g,r,p) == "C" or loader.getSuit(g,r,p) =="S":
                  inter_img.append((g,r,p))
    #%%

    suitsImg, suitsLabels = loader.getTrainTestSuits(inter_img)

    trainSuitsImg = suitsImg
    trainSuitsLabels = suitsLabels

    testSuitsImg = suitsImg[146:]
    testSuitsLabels = suitsLabels[146:]
    return trainSuitsImg, trainSuitsLabels, testSuitsImg, testSuitsLabels

