import cv2
import numpy as np
from glob import glob
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self):
        inputImgFolder = "./SR_dataset/T91_ILR"
        labelImgFolder = "./SR_dataset/T91_HR"
        patchSize = 32

        inputImgPaths = glob("%s/*.png" % (inputImgFolder))
        labelImgPaths = glob("%s/*.png" % (labelImgFolder))
        inputImgPaths.sort()
        labelImgPaths.sort()

        self.inputPatchs = []
        self.labelPatchs = []

        for idx in range(len(inputImgPaths)):
            inputImg = np.array(cv2.imread(inputImgPaths[idx]), dtype=np.float32) / 255.
            labelImg = np.array(cv2.imread(labelImgPaths[idx]), dtype=np.float32) / 255.

            inputImg = np.transpose(inputImg, [2, 0, 1])
            labelImg = np.transpose(labelImg, [2, 0, 1])

            self.frameToPatchs(inputImg=inputImg, labelImg=labelImg, patchSize=patchSize)

    def __len__(self):
        return len(self.inputPatchs)

    def __getitem__(self, idx):
        return self.inputPatchs[idx], self.labelPatchs[idx]

    def frameToPatchs(self, inputImg=None, labelImg=None, patchSize=32):
        channel, height, width = labelImg.shape

        numPatchY = height // patchSize
        numPatchX = width // patchSize

        for yIdx in range(numPatchY):
            for xIdx in range(numPatchX):
                xStartPos = xIdx * patchSize
                xFianlPos = (xIdx * patchSize) + patchSize
                yStartPos = yIdx * patchSize
                yFianlPos = (yIdx * patchSize) + patchSize

                self.inputPatchs.append(inputImg[:, yStartPos:yFianlPos, xStartPos:xFianlPos])
                self.labelPatchs.append(labelImg[:, yStartPos:yFianlPos, xStartPos:xFianlPos])


class TestDataset(Dataset):
    def __init__(self):
        inputImgFolder = "./SR_dataset/Set5_ILR"
        labelImgFolder = "./SR_dataset/Set5_HR"

        inputImgPaths = glob("%s/*.bmp" % (inputImgFolder))
        labelImgPaths = glob("%s/*.bmp" % (labelImgFolder))
        inputImgPaths.sort()
        labelImgPaths.sort()

        self.inputImgs = []
        self.labelImgs = []
        self.imgName = []

        for idx in range(len(inputImgPaths)):
            inputImg = np.array(cv2.imread(inputImgPaths[idx]), dtype=np.float32) / 255.
            labelImg = np.array(cv2.imread(labelImgPaths[idx]), dtype=np.float32) / 255.

            inputImg = np.transpose(inputImg, [2, 0, 1])
            labelImg = np.transpose(labelImg, [2, 0, 1])

            self.inputImgs.append(inputImg)
            self.labelImgs.append(labelImg)
            self.imgName.append(inputImgPaths[idx].split("\\")[-1])

    def __len__(self):
        return len(self.inputImgs)

    def __getitem__(self, idx):
        return self.inputImgs[idx], self.labelImgs[idx], self.imgName[idx]