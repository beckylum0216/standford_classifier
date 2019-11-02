import csv

import math
import os
import random

import cv2
from PIL import Image
from PictureHeader import PictureHeader
import numpy as np
import ast


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

    def Normalise(self, dirPath):
        normDir = "./normalised/"
        labelList = []
        fileList = os.walk(dirPath)
        for root, dirs, file in fileList:
            for ii in file:
                oldPath = os.path.join(dirPath, ii)
                dirp = os.path.split(os.path.dirname(dirPath))[1]
                #print(dirp)
                if os.path.isfile(oldPath):
                    newPath = os.path.join(normDir, "norm_"+ ii)
                    print(newPath)
                    img = Image.open(oldPath)
                    imgFile = cv2.equalizeHist(img)
                    width, height = imgFile.size
                    ratio = 100 / float(width)
                    dimensions = (100, int(height * ratio))
                    imgResize = imgFile.resize(dimensions, Image.ANTIALIAS)
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
        histolist = {}
        histofile = {}
        classmap = []

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
                        #print(len(row['file']))

                        histo[ii] = histo.get(ii, 0) + 1
                        #print(histo[ii])
                        filelist.append(row['file'])

                if filelist:
                    histolist[ii] = filelist

        lowest = 2000000
        classes = 0
        totclasses = 0


        for key, value in histo.items():
            #print("class: ", key, " count: ", value)
            if int(key) > classes:
                classes = int(key)

            if value >= 90:

                if lowest > value:
                    lowest = value


        labelfile.close()

        for key, value in histolist.items():
            # print("class: ", key, " file: ", len(value))
            if len(value) >= 90:
                if len(value) < 100:
                    classmap.append(key)
                    histofile[key] = value
                    totclasses += 1

        # for key, value in histofile.items():
        #     print("class: ", key, " file: ", len(value))
            # totclasses += 1
            #
            # if lowest > value:
            #     lowest = value

        print(lowest)
        print(classes)
        print(totclasses)

        return classes, lowest, totclasses, histofile, classmap

    def Pareto(self, histofile, dstpath, size):
        pareto80 = {}
        pareto20 = {}

        for key, value in histofile.items():
            temp = random.sample(set(value), size)
            size80 = int(size * 0.8)
            size20 = math.ceil(size * 0.2)
            #print("80:", size80, "20:", size20)
            pareto80[key] = temp[:size80]
            pareto20[key] = temp[-size20:]
            #print(pareto20)


        pareto80path = os.path.join(dstpath, "pareto80.par")
        with open(pareto80path, 'w+', newline='') as labelfile:
            wr = csv.writer(labelfile)
            for key, value in pareto80.items():
                wr.writerow([key, value])

        labelfile.close()

        pareto20path = os.path.join(dstpath, "pareto20.par")
        with open(pareto20path, 'w+', newline='') as labelfile:
            wr = csv.writer(labelfile)
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

    def OpenParetoFile(self, dirpath, filename, size):
        pareto = {}
        filepath = os.path.join(dirpath, filename)
        csv.register_dialect('par', delimiter=',', quoting=csv.QUOTE_NONE)
        with open(filepath, newline='\n') as labelfile:
            fields = ['class', 'filelist']
            csvreader = csv.DictReader(labelfile, fieldnames=fields)

            for ii in range(size):
                # print(ii)
                filelist = []
                labelfile.seek(0)
                for row in csvreader:

                    if int(row['class']) == ii:
                        filelist = ast.literal_eval(row['filelist'])
                        # print(histo[ii])
                        # print(filelist)

                if filelist:
                    pareto[ii] = filelist

        return pareto

    def OpenFiles(self, dirpath, filelist, classes, augment, map):
        imgList = []
        lblList = []
        originalList = []
        size =self.GetSize(dirpath, filelist)
        augmentsize = len(augment)
        ndImgList= np.empty((size * augmentsize, 100, 100, 1))
        ndLblList = np.empty((size * augmentsize, classes))
        count = 0
        for aa in augment:
            for key, value in filelist.items():
                for ii in value:
                    imgpath = os.path.join(dirpath, ii)
                    exists = os.path.exists(imgpath)
                    if exists:
                        imgfile = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
                        originalList.append(imgfile)
                        normalised = imgfile / 255
                        if aa == "Normalised":
                            normalised = imgfile / 255
                        elif aa == "Gabor":
                            gaborimg = self.ApplyGaborFilter(imgfile)
                            # ret, thresh = cv2.threshold(imgfile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            normalised = gaborimg / 255
                        elif aa == "Fourier":
                            fourier = self.ApplyFourier(imgfile)
                            normalised = fourier / 255
                        elif aa == "Horizontal":
                            horizontal = self.FlipHorizontal(imgfile)
                            normalised = horizontal / 255
                        elif aa == " Vertical":
                            vertical = self.FlipVertical(imgfile)
                            normalised = vertical / 255

                        temp = np.array(normalised)[:, :, np.newaxis]
                        #temp.flatten()
                        # print(temp)
                        #data = np.expand_dims(temp, axis=2)
                        ndImgList[count] = temp
                        imgList.append(imgfile)

                        ndLblList[count] = self.OneShotEncoder(classes, key, map)
                        index = 0
                        for ii in map:
                            if ii == key:
                                # print("ii: ", ii," key: ", key, " index: ", index)
                                lblList.append(index)
                            index += 1
                        count += 1

        return imgList, lblList, ndImgList, ndLblList, originalList, count

    def OpenFile(self, imgpath, classes, key, map):
        imgList = []
        lblList = []
        originalList = []
        size = 1
        ndImgList = np.empty((size, 100, 100, 1))
        ndLblList = np.empty((size, classes))
        count = 0

        exists = os.path.exists(imgpath)
        if exists:
            imgfile = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            originalList.append(imgfile)
            gaborimg = self.ApplyGaborFilter(imgfile)
            # ret, thresh = cv2.threshold(imgfile, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            normalised = gaborimg / 255
            temp = np.array(normalised)[:, :, np.newaxis]
            # temp.flatten()
            # print(temp)
            # data = np.expand_dims(temp, axis=2)
            ndImgList[count] = temp
            imgList.append(imgfile)
            lbl = int(key)
            ndLblList[count] = self.OneShotEncoder(classes, lbl, map)
            index = 0
            for ii in map:
                if ii == key:
                    lblList.append(index)
                index += 1
            count += 1

        return imgList, lblList, ndImgList, ndLblList, originalList, count


    def GetSize(self, dirpath, filelist):
        count = 0
        for key, value in filelist.items():
            for ii in value:
                imgpath = os.path.join(dirpath, ii)
                exists = os.path.exists(imgpath)
                if exists:
                    count += 1

        return count

    def OneShotEncoder(self, classes, key, map):
        count = 0
        encoder = np.zeros((1, classes))
        for ii in map:
            # print(ii)
            if key == ii:
                encoder[0, count] = 1
            else:
                count += 1

        # print(encoder)
        return encoder

    def GenerateBlackImg(self):
        black = np.zeros((100,100), dtype = np.float32)

        # cv2.imshow("black", black)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return black

    def GenerateWhiteImg(self):
        white = np.full((100, 100), 255, dtype = np.float32)

        # cv2.imshow("white", white)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return white

    def GenerateCalibrationImg(self, colour):
        img = np.full((100, 100), colour, dtype=np.float32)

        # cv2.imshow("calibration", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img

    def GenerateXORImg(self, colour):
        xor = np.empty((100,100), dtype = np.float32)
        for ii in range(100):
            for jj in range(100):
                if ii < 50 and jj < 50:
                    xor[ii][jj] = colour
                elif ii > 50 and jj < 50:
                    xor[ii][jj] = 0
                elif ii > 50 and jj > 50:
                    xor[ii][jj] = colour
                else:
                    xor[ii][jj] = 0

        # cv2.imshow("xor", xor)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return xor

    def GenerateTrainCalibrationImgs(self, N, nClasses):
        classes = nClasses
        size = N
        ndImgList = np.empty((size, 100, 100, 1))
        ndLblList = np.empty((size, classes))
        imgList = []
        lblList = []
        count = 0
        for ii in range(size):
            num = random.randrange(0, classes)

            if num == 0:
                img = self.GenerateWhiteImg()
                normalised = img/255
                temp = np.array(normalised)[:,:, np.newaxis]
                ndImgList[count] = temp
                ndLblList[count] = self.OneShotEncoder(classes, 0)
                lblList.append(0)
                count += 1
            elif num == 1:
                img = self.GenerateBlackImg()
                normalised = img / 255
                temp = np.array(normalised)[:, :, np.newaxis]
                ndImgList[count] = temp
                ndLblList[count] = self.OneShotEncoder(classes, 1)
                lblList.append(1)
                count += 1
            else:
                img = self.GenerateXORImg(255)
                normalised = img / 255
                temp = np.array(normalised)[:, :, np.newaxis]
                ndImgList[count] = temp
                ndLblList[count] = self.OneShotEncoder(classes, 2)
                lblList.append(2)
                count += 1

        imgList = ndImgList.tolist()

        return imgList, lblList, ndImgList, ndLblList

    def GenerateTestCalibrationImgs(self, N, nClasses):
        classes = nClasses
        size = N
        ndImgList = np.empty((size, 100, 100, 1))
        ndLblList = np.empty((size, classes))
        imgList = []
        lblList = []
        count = 0
        for ii in range(size):
            num = random.randrange(0, classes)

            if num == 0:
                img = self.GenerateCalibrationImg(200)
                normalised = img / 255
                temp = np.array(normalised)[:, :, np.newaxis]
                ndImgList[count] = temp
                ndLblList[count] = self.OneShotEncoder(classes, 0)
                lblList.append(0)
                count += 1
            elif num == 1:
                img = self.GenerateCalibrationImg(50)
                normalised = img / 255
                temp = np.array(normalised)[:, :, np.newaxis]
                ndImgList[count] = temp
                ndLblList[count] = self.OneShotEncoder(classes, 1)
                lblList.append(1)
                count += 1
            else:
                img = self.GenerateXORImg(200)
                normalised = img / 255
                temp = np.array(normalised)[:, :, np.newaxis]
                ndImgList[count] = temp
                ndLblList[count] = self.OneShotEncoder(classes, 2)
                lblList.append(2)
                count += 1

        imgList = ndImgList.tolist()

        return imgList, lblList, ndImgList, ndLblList


    def ApplyGaborFilter(self, targetImage):
        w, h = targetImage.shape
        gabor_filter = cv2.getGaborKernel((w, h), 5.0, np.pi/1, 13.0, 0.5, 0, ktype=cv2.CV_32F)
        gaborImg = cv2.filter2D(targetImage, cv2.CV_8UC3, gabor_filter)

        return gaborImg

    def PredictionHisto(self, y_prediction, classes):
        size = len(y_prediction)
        histo = {}
        histolist = np.zeros(classes)
        for ii in range(size):
            histo[y_prediction[ii]] = histo.get(y_prediction[ii], 0) + 1

        for key, value in histo.items():
            #print("class: ", key, " count: ", value)
            histolist[key] = value

        sortedlist = sorted(histo.items())
        for (key, value) in sortedlist:
            print("class: ", key, " count: ", value)

        return histolist


    def SaveConfusionMatrix(self, targetMatrix, filepath):
        matrixFile = open(filepath, "w")

        matrixFile.writelines(targetMatrix)

        matrixFile.close()

    def SaveClassification(self, targetReport, filepath):
        classFile = open(filepath, "w")
        classFile.writelines(targetReport)
        classFile.close()

    def ShowResult(self, targetImg, annotationfile, prediction, map):
        with open(annotationfile, 'r', newline='') as labelfile:
            lblList = []
            result = "blah"
            wr = csv.reader(labelfile, quoting=csv.QUOTE_NONE)
            for row in wr:
                lblList.append(row)

            temp = 0
            for ii in map:
                if ii == int(prediction):
                    temp = ii
                else:
                    temp += 1

            count = 0
            for ii in lblList:
                if count is int(temp):
                    result = str(ii[0])
                count += 1

            print(result)
            im = np.asmatrix(targetImg)
            font = cv2.FONT_HERSHEY_SIMPLEX
            finalImg = cv2.resize(im, (1080, 1080), interpolation=cv2.INTER_AREA)
            cv2.putText(finalImg, result, (10, 780), font, 3, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Result", finalImg)
            cv2.waitKey(5)
            cv2.destroyAllWindows()

            return result

    def ApplyFourier(self, targetImg):
        rows, cols =targetImg.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        dft = cv2.dft(np.float32(targetImg), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        mask = np.ones((rows, cols, 2), np.uint8)
        r = 5
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 1

        # apply mask and inverse DFT
        fshift = dft_shift * mask

        # fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # cv2.imshow("fourier", img_back)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img_back

    def FlipHorizontal(self, targetImg):
        horizontal = cv2.flip(targetImg, 0)

        # cv2.imshow("horizontal", horizontal)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return horizontal

    def FlipVertical(self, targetImg):
        vertical = cv2.flip(targetImg, 1)

        # cv2.imshow("vertical", vertical)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return vertical