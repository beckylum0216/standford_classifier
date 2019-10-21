import csv

import math
import os
import random

import cv2
from PIL import Image
from PictureHeader import PictureHeader
import numpy as np


class Utility:
    #leave for now
    def GetLabel(self, dir,oldPath, newPath,file):
        imgLabel = PictureHeader()

        imgLabel.imgOldPath = oldPath
        imgLabel.imgNewPath = newPath
        imgLabel.imgFile = file
        if dir == "Stanford_Dataset":
            imgLabel.imgType = "car"
            imgLabel.carBool = 1
        elif dir == "yalefaces":
            imgLabel.imgType = "face"
            imgLabel.carBool = 0
        else:
            print("Error: not in the list of dataset supported")

        return imgLabel

    def NormaliseSize(self, dirPath):
        normDir = "./normalised/"
        labelList = []
        fileList= os.walk(dirPath)
        for root, dirs, file in fileList:
            for ii in file:
                oldPath = os.path.join(dirPath, ii)
                dirp = os.path.split(os.path.dirname(dirPath))[1]
                #print(dirp)
                if os.path.isfile(oldPath):
                    newPath = os.path.join(normDir, "norm_"+ ii)
                    print(newPath)
                    imgFile = Image.open(oldPath)
                    width, height = imgFile.size
                    imgResize = imgFile.resize((100, 100), Image.ANTIALIAS)
                    imgResize.save(newPath, 'PNG', quality=90)
                    imgLabel = self.GetLabel(dirp, oldPath, newPath, ii)
                    labelList.append(imgLabel)

        # needs to be done on the fly while creating the file for keeping historical metadata
        with open("./detectionlabel/label.txt", 'w+', newline='') as labelfile:
            wr = csv.writer(labelfile, quoting=csv.QUOTE_ALL)
            for label in labelList:
                wr.writerow([label.imgOldPath, label.imgNewPath, label.imgFile, label.imgType, label.carBool])


    def StatCheck(self, file = "file path", nclasses=200):
        histo = {}
        histofile = {}

        csv.register_dialect('bbox', delimiter=',')
        with open(file, newline= '\n') as labelfile:
            fields = ['path', 'file', 'class', 'x1', 'y1', 'x2', 'y2']
            csvreader = csv.DictReader(labelfile, fieldnames=fields)

            for ii in range(nclasses):
                # print(ii)
                filelist = []
                labelfile.seek(0)
                for row in csvreader:
                    #print(int(row['class']), " ", ii)
                    if int(row['class']) == ii:
                        #print("true")
                        histo[ii] = histo.get(ii, 0) + 1
                        # print(histo[ii])
                        filelist.append(row['file'])

                if filelist:
                    histofile[ii] = filelist

        lowest = 2000000
        totclasses = 0


        for key, value in histo.items():
            print("class: ", key, " count: ", value)
            if totclasses < int(key):
                totclasses = int(key)

            if lowest > value:
                lowest = value

        print(lowest)
        print(totclasses)
        labelfile.close()

        # for key, value in histofile.items():
        #     print("class: ", key, " file: ", value)

        return totclasses, lowest, histofile

    def Pareto(self, histofile, dstpath, size):
        pareto80 = {}
        pareto20 = {}

        for key, value in histofile.items():
            temp = random.sample(set(value), size)
            size80 = int(size * 0.8)
            size20 = math.ceil(size * 0.2)
            #print("80:", size80, "20:", size20)
            pareto80[key] = temp[:40]
            pareto20[key] = temp[-8:]
            #print(pareto20)


        pareto80path = os.path.join(dstpath, "pareto80.csv")
        with open(pareto80path, 'w+', newline='') as labelfile:
            wr = csv.writer(labelfile, quoting=csv.QUOTE_ALL)
            for key, value in pareto80.items():
                wr.writerow([key, value])

        labelfile.close()

        pareto20path = os.path.join(dstpath, "pareto20.csv")
        with open(pareto20path, 'w+', newline='') as labelfile:
            wr = csv.writer(labelfile, quoting=csv.QUOTE_ALL)
            for key, value in pareto20.items():
                wr.writerow([key, value])

        labelfile.close()

        return pareto80, pareto20

    def McCall(self, histofile, classes = 0, lowest = 0, ):
        mccall70 = {}
        mccall20 = {}
        mccall10 = {}

        for key, value in histofile.items():
            temp = random.sample(set(value), 48)
            mccall70[key] = temp[0:34]
            mccall20[key] = temp[34:43]
            mccall10[key] = temp[43:48]

    def OpenParetoFile(self, dirpath, filename):
        print("blah")

    def OpenFiles(self, dirpath, filelist, classes):

        imgList = []
        lblList = []
        size =self.GetSize(dirpath, filelist)
        ndImgList= np.empty((size, 100, 100, 1))
        ndLblList = np.empty((size, classes))
        count = 0
        for key, value in filelist.items():
            for ii in value:
                imgpath = os.path.join(dirpath, ii)
                exists = os.path.exists(imgpath)
                if exists:
                    imgfile = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
                    temp = np.array(imgfile)[:, :, np.newaxis]
                    #temp.flatten()
                    # print(temp)
                    #data = np.expand_dims(temp, axis=2)
                    ndImgList[count] = temp
                    imgList.append(imgfile)

                    ndLblList[count] = self.OneShotEncoder(classes, key)
                    lblList.append(key)
                    count += 1

        return imgList, lblList, ndImgList, ndLblList, count


    def GetSize(self, dirpath, filelist):
        count = 0
        for key, value in filelist.items():
            for ii in value:
                imgpath = os.path.join(dirpath, ii)
                exists = os.path.exists(imgpath)
                if exists:
                    count += 1

        return count

    def OneShotEncoder(self, classes, key):
        encoder = np.zeros((1, classes))
        encoder[0, key - 1] = 1

        return encoder

