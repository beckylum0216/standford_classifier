import csv
import os
from PIL import Image
import cv2
import numpy as np
from skimage import feature

class ImageDetector(object):
    bbox = {}

    def GetCentroid(self, x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)

        cx = x + x1
        cy = y + y1

        return (cx, cy)

    def DetectVehicles(self, fg_mask, min_contour_width=35, min_contour_height=35):

        matches = []
        roi = []
        # finding external contours
        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        # filtering by with, height
        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_valid = (w >= min_contour_width) and (
                    h >= min_contour_height)

            if not contour_valid:
                continue

            # getting center of the bounding box
            centroid = self.GetCentroid(x, y, w, h)

            matches.append(((x, y, w, h), centroid))

            roi.append((x,y,w,h))

        return roi

    def GetPicture(self, path="blah"):
        print(path)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), sigmaX=0)
        threshhold, threshholdImg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(threshhold)
        cv2.imshow('Thresh', threshholdImg)
        # cv2.waitKey(0)

        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(threshholdImg, cv2.MORPH_OPEN, kernel, iterations=2)

        gabor_filter = cv2.getGaborKernel((25, 25), 2.0, np.pi / 8, 20.0, 0.5, 0, ktype=cv2.CV_32F)
        gaborImg = cv2.filter2D(morph, cv2.CV_8UC3, gabor_filter)

        cv2.imshow('Gabor', gaborImg)

        canny = cv2.Canny(gaborImg, 100, 300)
        cv2.imshow('Canny', canny)
        # cv2.waitKey(0)


        # dir, file = path.split('_')
        # print(file)
        # gaborPath = "gabor/" + file
        # cv2.imwrite(gaborPath, gaborImg)

        #this section is just for checking if the bounded box has been correctly placed - not needed anymore
        boundedcar = self.DetectVehicles(canny, 5, 5)
        for (x, y, w, h) in boundedcar:
            selected = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print("x: ", x, " y: ", y, " w: ", w, " h: ", h)


        root, dir, sub, sub1 = path.split('/')
        directory, id = str(sub1).split('\\')
        rows = self.bbox[id]

        for data in rows:
            print("ground truth x:", data['ngx1'], " y: ", data['ngy1'], " w: ", data['ngx2'], " h:", data['ngy2'])
            selected = cv2.rectangle(image,(int(data['x1']), int(data['y1'])), (int(data['x2']), int(data['y2'])), (0, 255, 255), 2)

        # the window showing output image with corners
        cv2.imshow('Image with Borders', image)
        cv2.resizeWindow('image with borders', 100, 100)
        cv2.waitKey(0)


    def SubtractBackground(self, path ="path", outPath ="path"):
        fileList = os.walk(path)
        for root, dir, file in fileList:
            for ii in file:
                filePath = os.path.join(path, ii)
                self.GetPicture(filePath)

    def GetStanfordBBox(self, imgSize, annotationpath="annotationPath"):
        with open(annotationpath, newline='\n') as csvfile:
            csvInput = csv.DictReader(csvfile, delimiter=',')
            for rows in csvInput:
                data = {}
                line = []
                carPicture = Image.open(rows['path'])
                width, height = carPicture.size

                dir, id = str(rows['path']).split('/')
                #print(id)
                data['path'] = rows['path']
                data['file'] = id

                data['class'] = rows['class']
                data['x1'] = rows['x1']
                data['y1'] = rows['y1']
                data['x2'] = rows['x2']
                data['y2'] = rows['y2']

                newWidth = imgSize/width
                newHeight = imgSize/height

                data['ngx1'] = int(int(rows['x1']) * newWidth)
                data['ngy1'] = int(int(rows['y1']) * newHeight)
                data['ngx2'] = int(int(rows['x2']) * newWidth)
                data['ngy2'] = int(int(rows['y2']) * newHeight)
                line.append(data)
                self.bbox[id] = line


        with open("./detectionlabel/normbbox.txt", 'w+', newline='') as labelfile:
            wr = csv.writer(labelfile, quoting=csv.QUOTE_ALL)
            for key in self.bbox.keys():
                for rows in self.bbox[key]:
                    wr.writerow([self.bbox[key][rows]['path'], self.bbox[key][rows]['class'],
                                 self.bbox[key][rows]['x1'],self.bbox[key][rows]['y1'],
                                 self.bbox[key][rows]['x2'], self.bbox[key][rows]['y2'],
                                 self.bbox[key][rows]['ngx1'], self.bbox[key][rows]['ngy1'],
                                 self.bbox[key][rows]['ngx2'], self.bbox[key][rows]['ngy2']])
        labelfile.close()

        return self.bbox

    def GetKittiBBox(self, newImageSize = 0, annotatePath = "path", imagePath = "path"):
        lblList = os.walk(annotatePath)
        for root, dir, file in lblList:
            #print(file)
            for ii in file:
                filepath = os.path.join(annotatePath, ii)
                #print(filepath)
                with open(filepath, newline="\n") as txtfile:
                    line = []
                    txtInput = csv.reader(txtfile, delimiter = ' ')
                    for rows in txtInput:
                        data = {}

                        name, extension = str(ii).split('.')
                        fileName = name +".png"
                        imgFile = os.path.join(imagePath, fileName)
                        #print(imgFile)
                        carpic = Image.open(imgFile)
                        width, height = carpic.size
                        data['path'] = fileName
                        data['class'] = self.EnumerateClass(rows[0])
                        data['x1'] = int(float(rows[4]))
                        data['y1'] = int(float(rows[5]))
                        data['x2'] = int(float(rows[6]))
                        data['y2'] = int(float(rows[7]))

                        newWidth = newImageSize / width
                        newHeight = newImageSize / height

                        data['ngx1'] = int(float(rows[4]) * newWidth)
                        data['ngy1'] = int(float(rows[5]) * newHeight)
                        data['ngx2'] = int(float(rows[6]) * newWidth)
                        data['ngy2'] = int(float(rows[7]) * newHeight)
                        line.append(data)
                    self.bbox[fileName] = line

        with open("./detectionlabel/kittibbox.txt", 'w+', newline='') as labelfile:
            wr = csv.writer(labelfile, quoting=csv.QUOTE_ALL)
            for key in self.bbox.keys():
                # print("list?" +  str(self.bbox[key]))
                for rows in self.bbox[key]:

                    wr.writerow([rows['path'], rows['class'],
                                 rows['x1'], rows['y1'],
                                 rows['x2'], rows['y2'],
                                 rows['ngx1'], rows['ngy1'],
                                 rows['ngx2'], rows['ngy2']])

        return self.bbox

    def EnumerateClass(self, inputString):
        case = {'Car':1 , 'Van':2, 'Truck':3,
                 'Pedestrian':4, 'Person_sitting':5, 'Cyclist':6,
                 'Tram':7, 'Misc':8, 'DontCare':9}

        return case[inputString]

    def LoadBBox(self, filepath = "path"):
        with open(filepath, 'r', newline='') as labelfile:
            fileName = "blah"

            line = []
            txtInput = csv.reader(labelfile, delimiter=',')
            for rows in txtInput:
                fileName = str(rows[0])
                #print("loading? "+ fileName)
                data = {}

                data['path'] = rows[0]
                data['class'] = rows[1]
                data['x1'] = int(rows[2])
                data['y1'] = int(rows[3])
                data['x2'] = int(rows[4])
                data['y2'] = int(rows[5])
                data['ngx1'] = int(rows[6])
                data['ngy1'] = int(rows[7])
                data['ngx2'] = int(rows[8])
                data['ngy2'] = int(rows[9])
                line.append(data)
                self.bbox[fileName] = line
        #print(self.bbox)