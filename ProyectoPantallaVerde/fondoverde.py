import matplotlib.pyplot as plt
import path
from PIL import Image
import numpy as np
import cv2
import scipy
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn import linear_model
from collections import Counter
import math
import copy
import os.path
import pickle

class BackClassifier:
    def __init__(self):
        self.testX = []
        self.testY = []
        self.arrX = []
        self.arrY = []
        self.rowsX = 0
        self.rowsY = 0
        self.colsX = 0
        self.colsY = 0
        self.chanX = 0
        self.chanY = 0
        self.eleX = 0
        self.eleY = 0
        self.bacX = []
        self.objX = []
        self.backValue = 0
        self.objectValue = 255
        self.sampleX = []
        self.sampleY = []
    def _arrange(self, img):
        if len(img.shape) == 3:
            rows, cols, chan = img.shape
        elif len(img.shape) == 2:
            rows, cols = img.shape
            chan = 1
        else:
            raise ValueException('Array of image has an invalid shape.')
        if (chan > 1):
            arr = np.reshape(img,(rows*cols, chan))
        else:
            arr = np.reshape(img,(rows*cols,))
        return arr, rows, cols, chan

    def _arrangeMany(imgs):
        if (len(img.shape) == 4):
            ele, rows, cols ,chan = img.shape
        elif (len(img.shape) == 3):
            ele, rows, cols = img.shape
            chan = 1
        else:
            raise ValueException('Array of images has an invalid shape.')
        if (chan > 1):
            arr = np.reshape(img,(ele*rows*cols, chan))
        else:
            arr = np.reshape(img,(ele*rows*cols,))
        return arr, ele, rows, cols, chan

    def _prepare(self, testX, testY, backValue, objectValue, multiple = False):
        self.testX = testX
        self.testY = testY
        if (multiple):
            self.arrX, self.eleX, self.rowsX, self.colsX, self.chanX = self._arrangeMany(testX)
            self.arrY, self.eleY, self.rowsY, self.colsY, self.chanY = self._arrangeMany(testY)
        else:
            self.arrX, self.rowsX, self.colsX, self.chanX = self._arrange(testX)
            self.arrY, self.rowsY, self.colsY, self.chanY = self._arrange(testY)
            self.eleX = 1
            self.eleY = 1
        self.backValue = backValue
        self.objectValue = objectValue
        self.bacX = self.arrX[self.arrY == backValue,:]
        self.objX = self.arrX[self.arrY == objectValue,:]

    def _fitOperation(self):
        pass

    def _predictOperation(self, X):
        pass

    def fit(self, testX, testY, backValue = 0, objectValue = 255, multiple = False):
        self._prepare(testX, testY, backValue, objectValue, multiple)
        self._fitOperation()

    def predict(self, img):
        arrTestX, rows, cols, chan = self._arrange(img)
        ans = self._predictOperation(arrTestX)
        ans = np.reshape(ans, (rows, cols))
        return ans

    def _makeSamples(self):
        self.sampleX = copy.deepcopy(self.objX)
        self.sampleX = self.sampleX.tolist()
        b = copy.deepcopy(self.bacX)
        for p in b:
            self.sampleX.append(p)
        self.sampleX = np.array(self.sampleX)
        a = np.full((len(self.objX),),self.objectValue,dtype = np.uint8)
        b = np.full((len(self.bacX)), self.backValue,dtype = np.uint8)
        self.sampleY = np.append(a, b)

class LogisticRegBackClassifier(BackClassifier):

    def __init__(self):
        BackClassifier.__init__(self)
        self.model = linear_model.LogisticRegression()

    def _fitOperation(self):
        self._makeSamples()
        self.model.fit(self.sampleX,self.sampleY)

    def _predictOperation(self, X):
        return self.model.predict(X)


class SupportVectorMachineBackClassifier(BackClassifier):

    def __init__(self):
        BackClassifier.__init__(self)
        self.model = svm.LinearSVC()

    def _fitOperation(self):
        self._makeSamples()
        self.model.fit(self.sampleX, self.sampleY)

    def _predictOperation(self, X):
        return self.model.predict(X)

def edit(img, mask, reyes, backValue = 0, objectValue = 255):
    #img[mask[:,:] == backValue] = [255, 0, 0]
    a = mask[:,:] == backValue
    b = mask[:,:] == objectValue
    img[a] = [0,0,0]
    temp = copy.deepcopy(reyes)
    temp[b] = [0,0,0]
    img = img + temp
    return img

filename = 'saved.bin'


def setClassifier():
    if os.path.isfile(filename):
        return pickle.load( open(filename, 'rb'))
    else:
        Xtest = cv2.imread('testX.png', cv2.IMREAD_COLOR)
        #Xtest = cv2.cvtColor(Xtest, cv2.COLOR_BGR2HSV)
        Ytest = cv2.imread('testY.png', cv2.IMREAD_GRAYSCALE)
        sv = SupportVectorMachineBackClassifier()
        sv.fit(Xtest, Ytest)
        pickle.dump(sv, open(filename, 'wb'))
        return sv

sv = setClassifier()

cap = cv2.VideoCapture(0)

first = False
while (True):
    ret, frame = cap.read()

    if not frame is None:
        if (not first):
            rows, cols, chan = frame.shape
            reyes = np.zeros((rows,cols,chan),dtype = np.uint8)
            reyes[:,:] = [255,0,0]
            first = True
        if (cv2.waitKey(1) & 0xFF == ord('a')):
            reyes = copy.deepcopy(frame)
            first = True
        #converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = sv.predict(frame)
        frame = edit(frame, mask, reyes)
        cv2.imshow('python plz',frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()

cv2.destroyAllWindows()
