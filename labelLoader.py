# %%
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


# %%
class Loader():
    def __init__(self, pathCSVbase):

        l = []
        for i in range(1, 8):
            # pathCSVbase = "C:/Users/maxim/Google Drive/Epfl/MA4/Img analysis/project/iapr/project/train_games/"
            pathCSV = f"{pathCSVbase}/game{i}/game{i}.csv"
            df = pd.read_csv(pathCSV)
            l.append(df)

            cardsLabel = pd.concat(l, keys=['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7'], axis=0).reset_index(level=1)
        self.df = cardsLabel

    def getNumLabel(self, g, r, p):
        tempStr = self.df.loc[f'g{g}'][f'P{p}'][r - 1][0]

        # 0-9
        if tempStr in ["K", "Q", "J"]:
            return 10
        else:
            return int(tempStr)

    def getSuit(self, g, r, p):
        return self.df.loc[f'g{g}'][f'P{p}'][r - 1][1]

    def getSmallNumber(self, g, r, p, plotFlag=False):

        pathIm = f"numbers/game_{g}_round_{r}_player_{p}.png"

        if os.path.isfile(pathIm):
            im = cv.imread(pathIm)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if plotFlag:
            plt.imshow(gray)
            plt.show()

        histo, imgBin = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        if plotFlag:
            plt.imshow(imgBin)
            plt.show()
            # plt.hist(histo.ravel(), 256)
            # plt.show()
        squarred = np.pad(imgBin, ((0, 0), (5, 5)), 'constant', constant_values=(255))
        squarred = cv2.bitwise_not(squarred)
        smallImg = cv2.resize(squarred, (28, 28))
        if plotFlag:
            plt.imshow(squarred, cmap="gray")
            plt.show()

            plt.imshow(smallImg, cmap="gray")
            plt.show()
        return smallImg

    def getNumberAndLabel(self, g, r, p):
        return self.getSmallNumber(g, r, p), self.getNumLabel(g, r, p)

    def getTrain(self):
        nP = 4
        nR = 13
        nG = 6
        trainlabels = torch.empty((nG, nR, nP))
        trainImg = torch.empty((nG, nR, nP, 1, 28, 28))



        for g in range(1, nG + 1):
            for r in range(1, nR + 1):
                for p in range(1, nP + 1):
                    img, lab = self.getNumberAndLabel(g, r, p)
                    trainImg[g-1, r-1, p-1, 0, :, :] = torch.from_numpy(img/255)
                    trainlabels[g-1, r-1, p-1] = lab

        self.mean = trainImg.mean()
        self.std = trainImg.std()

        trainImg.sub_(self.mean).div_(self.std)
        return trainImg, trainlabels

    def getTest(self):

        TEST_SET = 7
        nP = 4
        nR = 13

        testLabels = torch.empty((nR, nP))
        testImg = torch.empty((nR, nP, 1, 28, 28))

        for r in range(1, nR + 1):
            for p in range(1, nP + 1):
                img, lab = self.getNumberAndLabel(TEST_SET, r, p)
                testImg[r-1, p-1, 0, :, :] = torch.from_numpy(img/255)
                testLabels[r-1, p-1] = lab
        mean = 0.1755
        std = 0.3720
        testImg.sub_(mean).div_(std)
        return testImg, testLabels

    def getExam(self):
        #a
        EXAM_SET = 8 # put 8 when ready

        nP = 4
        nR = 13

        testImg = torch.empty((nR, nP, 1, 28, 28))

        for r in range(1, nR + 1):
            for p in range(1, nP + 1):
                img = self.getSmallNumber(EXAM_SET, r, p)
                testImg[r-1, p-1, 0, :, :] = torch.from_numpy(img/255)

        mean = 0.1755
        std = 0.3720
        testImg.sub_(mean).div_(std)
        return testImg


if __name__ == "__main__":
    loader = Loader("C:/Users/maxim/Google Drive/Epfl/MA4/Img analysis/project/iapr/project/train_games/")
    loader.getTrain()