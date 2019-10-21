import os
import numpy as np

from CarCNN import NeuralNet
from CarDetector import CarDetector
from PictureHeader import PictureHeader
from Utility import Utility
#from CarMxNet import NeuralNet as MXCNN
from ImagePreprocess import ImageDetector
from CarKNN import NearestNeighbour
#from CarClassifier import NeuralNet
from harrisdetector import Detector



def main():
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
    savedfile = "savedmodels/stanford"

    imgTrain = []
    lblTrain = []
    imgTest = []
    lblTest = []
    labelList = []
    imageList = []


    # haar.GetKittiBBox(100, labels, images)
    #kitti = "./detectionlabel/kittibbox.txt"
    # haar.LoadBBox(kitti)
    # haar.SubtractBackground(images)


    #harris.Subtract_BG(path)
    # path = []
    # path.append("../Stanford_Dataset/car_ims")
    # path.append("./yalefaces/faces")
    # for ii in path:
    #     labelList = ut.NormaliseSize(ii)
    #imgpath = "./KITTI_Dataset/training/images/000006.png"
    # cd.run_dnn(imgpath, normpath)

    # haar.GetStanfordBBox(100, stanfordannotation)

    classes, lowest, histo =ut.StatCheck(annotationpath, 200)
    pareto80, pareto20 = ut.Pareto(histo, paretopath, lowest)
    knnImgTrain, knnLblTrain, cnnImgTrain, cnnLblTrain, pop80 = ut.OpenFiles(normpath, pareto80, classes)
    knnImgTest, knnLblTest, cnnImgTest, cnnLblTest, pop20 = ut.OpenFiles(normpath, pareto20, classes)
    #knn.RunKNN(lblTest, imgTest, lblTrain, imgTrain, 13)
    # #net = NeuralNet(cnnImgTest, cnnLblTest, cnnImgTrain, cnnLblTrain, 3, savedpath)
    # lblshape = cnnLblTrain.shape
    # lblsize = cnnLblTrain.size
    # print("size", lblsize, "shape", lblshape)
    cnn = NeuralNet(100, 100, classes, pop80, pop20, classes)

    cnn.CNeuralNet(cnnImgTrain, cnnLblTrain, cnnImgTest, cnnLblTest, 800, savedfile, 32)



if __name__ == '__main__':
    main()