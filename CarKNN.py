import csv
import math
import sys

import cv2
import numpy as np
import time

MAXSIZE = sys.maxsize

class NearestNeighbour(object):

    def __init__(self):
        self.acclist = []

    def EuclideanDistance(self, trainImg, testImg, size):
        distance = 0
        result = MAXSIZE
        imgtrain= np.append(trainImg, [])
        imgtest = np.append(testImg, [])

        imgtrain.flatten()
        imgtest.flatten()

        for ii in range(size):
            if imgtest.any() is not None:
                distance += int(pow(imgtest[ii]-imgtrain[ii], 2))
                #print("pixel: ", ii, "train type: ", imgtrain[ii], " test type: ", imgtest[ii])
                # print("l2: ", distance)
                result = math.sqrt(distance)
            # else:
                #print("Null pixels!!!")


        return result

    def KNearestNeighbour(self, trainLbl, trainImg, testImg, k = 0):
        if trainImg is None:
            trainImg = [[]]

        # if type(testImg) is None:
        #     testImg = []

        distances = {}
        neighbours = []
        if testImg is None:
            print("Motor Vehicle not detected!!!")
        else:
            size = len(testImg)
            count = 0
            for ii in trainImg:
                l2distance = self.EuclideanDistance(testImg, ii,size)
                distances[trainLbl[count]] = l2distance
                count += 1
            #should return a list of sorted keys
            sorteddist = sorted(distances, key=distances.get, reverse=True)

            # for ii in sorteddist:
            #     print("sorted: ", ii)

            for ii in range(k):
                neighbours.append((sorteddist)[ii])

            # for key, value in neighbours.items():
            #     print("neighbours key: ", key, " value: ", value)


        return neighbours

    def KNNPrediction(self, neighbours):
        votes = {-1:-1}
        for ii in range(len(neighbours)):
            response = neighbours[ii]
            if response in votes:
                votes[response] += 1
            else:
                votes[response] = 1

        sortedvotes = sorted(votes, reverse=True)
        result = list(sortedvotes)[0]
        # print("result: ", result)
        return result

    def KNNEvaluation(self, resultlabels, predictions):
        correct = 0
        count = 0
        for ii in resultlabels:
            if ii == predictions[count]:
                correct += 1
            count += 1
        return (correct / len(resultlabels)) * 100.00

    def RunKNN(self, testLbl, testImg, trainLbl, trainImg, k):
        predictions = []
        count = 0
        for ii in testImg:
            #print("testimg:" , ii)
            neighbours = self.KNearestNeighbour(trainLbl, trainImg, ii, k)
            result = self.KNNPrediction(neighbours)
            predictions.append(result)
            count += 1
            print("test img: ", count, " result: ", result)

        accuracy = self.KNNEvaluation(testLbl, predictions)
        print("Accuracy: ", accuracy, "%")
        self.acclist.append(accuracy)

    def KNNPredict(self, targetImg, trainLbl, trainImg, k):
        neighbours = self.KNearestNeighbour(trainLbl, trainImg, targetImg, k)
        result = self.KNNPrediction(neighbours)
        print("Result:", result)


    def SaveResultKNN(self, savepath, k):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filepath = savepath + "/KNN_Results_Log_" + str(k) + "_" + str(timestr) + ".log"
        with open(filepath, 'w+', newline='') as savefile:
            wr = csv.writer(savefile)
            wr.writerow(self.acclist)
        savefile.close()





