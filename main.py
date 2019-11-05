#!/usr/bin/python

import os
import numpy as np
import sys
import argparse
from CarCNN import NeuralNet
from CarDetector import CarDetector
from PictureHeader import PictureHeader
from Utility import Utility
#from CarMxNet import NeuralNet as MXCNN
from ImagePreprocess import ImageDetector
from CarKNN import NearestNeighbour
#from CarClassifier import NeuralNet
from harrisdetector import Detector



def run(argv):
    knn = NearestNeighbour()
    haar = ImageDetector()
    ut = Utility()
    cd = CarDetector()

    harris = Detector()
    haar = ImageDetector()
    ut = Utility()

    stanfordannotation = "./car_labels/annotations.txt"
    annotation = "../KITTI_Dataset/training/label_2"
    imagePath = "../KITTI_Dataset/training/image_2"
    labels = "./KITTI_Dataset/training/labels"
    images = "./KITTI_Dataset/training/images"
    imgpath = "./car_ims"
    normpath = "normalised"
    annotationpath = "./detectionlabel/normbbox.bbox"
    paretopath = "pareto"
    # savedfile = "savedmodels/stanford"
    # savedfile = "savedmodels/allaugment"
    savedfile = "savedmodels/basewithflips"
    calibrationsave = "calibrationsave/calibrate"
    resultpath = "results"
    imgTrain = []
    lblTrain = []
    imgTest = []
    lblTest = []
    labelList = []
    imageList = []

    # haar.GetKittiBBox(100, labels, images)
    #kitti = "./detectionlabel/kittibbox.txt"
    # haar.LoadBBox(kitti)

    # normalising dataset using ground truth bounding box might be too clean
    # haar.GetStanfordBBox(100, stanfordannotation)
    # haar.SubtractBackground(imgpath, normpath)

    # cd.run_dnn(imgpath, normpath)

    # for ii in range(1, 17):
    #     knn.RunKNN(knnLblTest, knnImgTest, knnLblTrain, knnImgTrain, ii)
    #
    # knn.SaveResultKNN(resultpath, ii)

    classes, lowest, totclasses, histo, map = ut.StatCheck(annotationpath, 200)
    # lowest = 25

    # print(pareto20)

    # pareto80file = "pareto80.par"
    # pareto80 = ut.OpenParetoFile(paretopath, pareto80file, classes)
    # pareto20file = "pareto20.par"
    # pareto20 = ut.OpenParetoFile(paretopath, pareto20file, classes)

    # cnn.TrainCNN(cnnImgTrain, cnnLblTrain, cnnImgTest, cnnLblTest, 1600, savedfile, 32)


    # cnn.LoadCNN(savedfile, pop20, cnnImgTest, cnnLblTest, 32)



    if argv.NN == "knn":
        augment = ["Normalised", "Gabor", "Canny", "Fourier", "Horizontal", "Vertical"]
        pareto80file = "pareto80.par"
        pareto80 = ut.OpenParetoFile(paretopath, pareto80file, classes)
        knnImgTrain, knnLblTrain, cnnImgTrain, cnnLblTrain, origin, pop80 = ut.OpenFiles(normpath, pareto80, totclasses, augment, map)
        knnImgTest, knnLblTest, cnnImgTest, cnnLblTest, origin, pop20 = ut.OpenFile(argv.target, classes, argv.label, augment, map)
        knn.KNNPredict(knnImgTest, knnLblTrain, knnImgTrain, 3)
    elif argv.NN == "cnn":
        augment = ["Normalised", "Gabor", "Fourier", "Horizontal", "Vertical"]
        pareto80file = "pareto80.par"
        pareto80 = ut.OpenParetoFile(paretopath, pareto80file, totclasses)
        knnImgTrain, knnLblTrain, cnnImgTrain, cnnLblTrain, origin, pop80 = ut.OpenFiles(normpath, pareto80, totclasses, augment, map)
        knnImgTest, knnLblTest, cnnImgTest, cnnLblTest, origin, pop20 = ut.OpenFile(argv.target, totclasses, argv.label, map)
        # print("encoded label", cnnLblTest.argmax())
        cnn = NeuralNet(100, 100, totclasses, pop80, pop20)
        cnn.PredictCNN(savedfile, pop20, cnnImgTest, cnnLblTest, origin, 32, map)
    elif argv.NN == "random":
        augment = ["Canny"]
        # augment = ["Normalised", "Gabor", "Fourier", "Horizontal", "Vertical", "Canny"]
        pareto80file = "pareto80.par"
        pareto80 = ut.OpenParetoFile(paretopath, pareto80file, classes)
        pareto20file = "pareto20.par"
        pareto20 = ut.OpenParetoFile(paretopath, pareto20file, classes)
        knnImgTrain, knnLblTrain, cnnImgTrain, cnnLblTrain, origin, pop80 = ut.OpenFiles(normpath, pareto80, totclasses, augment, map)
        knnImgTest, knnLblTest, cnnImgTest, cnnLblTest, origin,pop20 = ut.OpenFiles(normpath, pareto20, totclasses, augment, map)
        cnn = NeuralNet(100, 100, totclasses, pop80, pop20)
        cnn.LoadCNN(savedfile, pop20, cnnImgTest, cnnLblTest, origin, 32, map)
    elif argv.NN == "train":
        augment = ["Normalised"]
        pareto80, pareto20 = ut.Pareto(histo, paretopath, lowest)
        knnImgTrain, knnLblTrain, cnnImgTrain, cnnLblTrain, origin, pop80 = ut.OpenFiles(normpath, pareto80, totclasses, augment, map)
        knnImgTest, knnLblTest, cnnImgTest, cnnLblTest, origin, pop20 = ut.OpenFiles(normpath, pareto20, totclasses, augment, map)
        cnn = NeuralNet(100, 100, totclasses, pop80, pop20)
        cnn.TrainCNN(cnnImgTrain, cnnLblTrain, cnnImgTest, cnnLblTest, 900, savedfile, origin, 32, map)
    elif argv.NN == "calibrate":
        classes = 3
        pop80 = 500
        pop20 = 100
        knnImgTrain, knnLblTrain, cnnImgTrain, cnnLblTrain = ut.GenerateTrainCalibrationImgs(pop80, classes)
        knnImgTest, knnLblTest, cnnImgTest, cnnLblTest = ut.GenerateTestCalibrationImgs(pop20, classes)
        cnn = NeuralNet(100, 100, classes, pop80, pop20)
        cnn.TrainCNN(cnnImgTrain, cnnLblTrain, cnnImgTest, cnnLblTest, 200, calibrationsave, cnnImgTest, 32, map)
        # cnn.LoadCNN(calibrationsave, pop20, cnnImgTest, cnnLblTest, 32)
    elif argv.NN == "kmean":
        augment = ["Normalised", "Gabor", "Canny", "Fourier", "Horizontal", "Vertical"]
        pareto80file = "pareto80.par"
        pareto80 = ut.OpenParetoFile(paretopath, pareto80file, classes)
        pareto20file = "pareto20.par"
        pareto20 = ut.OpenParetoFile(paretopath, pareto20file, classes)
        knnImgTrain, knnLblTrain, cnnImgTrain, cnnLblTrain, origin, pop80 = ut.OpenFiles(normpath, pareto80, totclasses, augment, map)
        knnImgTest, knnLblTest, cnnImgTest, cnnLblTest, origin, pop20 = ut.OpenFiles(normpath, pareto20, totclasses, augment, map)
        knn.RunKNN(knnLblTest, knnImgTest, knnLblTrain, knnImgTrain, 5)


def main():
    parser = argparse.ArgumentParser(description= "CarClassifier")
    if len(sys.argv) < 4:
        parser.add_argument('--nn', '-n', dest='NN', choices=['random', 'train', 'calibrate', "kmean"], help="choose classification method")
    else:
        parser.add_argument('--nn', '-n', dest='NN', choices=['knn', 'cnn'], help="choose classification method")
        parser.add_argument('target', help="the filepath or file name")
        parser.add_argument('label', help="the target label")

    argv = parser.parse_args()
    run(argv)

def blah():
    normpath = "normalised"
    annotationpath = "./detectionlabel/normbbox.bbox"
    paretopath = "pareto"
    savedfile = "savedmodels/stanford"
    calibrationsave = "calibrationsave/calibrate"
    augment = ["Canny"]
    ut = Utility()
    classes, lowest, totclasses, histo, map = ut.StatCheck(annotationpath, 200)
    pareto80file = "pareto80.par"
    pareto80 = ut.OpenParetoFile(paretopath, pareto80file, classes)
    pareto20file = "pareto20.par"
    pareto20 = ut.OpenParetoFile(paretopath, pareto20file, classes)
    knnImgTrain, knnLblTrain, cnnImgTrain, cnnLblTrain, origin, pop80 = ut.OpenFiles(normpath, pareto80, totclasses, augment, map)
    knnImgTest, knnLblTest, cnnImgTest, cnnLblTest, origin, pop20 = ut.OpenFiles(normpath, pareto20, totclasses, augment, map)

if __name__ == '__main__':
    main()
    # blah()