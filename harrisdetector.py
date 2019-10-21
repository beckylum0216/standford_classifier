import math
import os
from struct import *
import cv2
import numpy as np

from PictureHeader import PictureHeader


class Detector(object):

    def Preprocessor(self, picture):
        image = cv2.imread(picture)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mog = cv2.createBackgroundSubtractorMOG2()
        foreground = mog.apply(gray)
        canny = cv2.Canny(foreground, 100, 200)



        # apply the cv2.cornerHarris method
        # to detect the corners with appropriate
        # values as input parameters
        # dest = cv2.cornerHarris(canny, 2, 5, 0.07)

        # Results are marked through the dilated corners
        dest = cv2.dilate(canny, None)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        cv2.imshow('background', sure_bg)
        cv2.waitKey(0)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # cv2.imshow('foreground', sure_fg)
        # cv2.waitKey(0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # cv2.imshow('unknown', unknown)
        # cv2.waitKey(0)

        # # Marker labelling
        # ret, markers = cv2.connectedComponents(sure_fg)
        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers + 1
        # # Now, mark the region of unknown with zero
        # markers[unknown == 255] = 0
        #
        # markers = cv2.watershed(image, markers)
        # image[markers == -1] = [255, 255, 0]

        # the window showing output image with corners
        cv2.imshow('Image with Borders', dest)
        cv2.waitKey(0)

    def Subtract_BG(self, path = "path"):
        fileList = os.walk(path)
        for root, dir, file in fileList:
            for ii in file:
                filePath = os.path.join(path, ii)
                self.Preprocessor(filePath)


