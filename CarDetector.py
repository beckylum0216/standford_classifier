
import os
import cv2
from PIL import Image
import numpy as np
import Utility as ut

class CarDetector(object):

    argWeight = "./YOLO/yolov3.weights"
    argConfig = "./YOLO/yolov3.cfg"
    argClass = "./YOLO/yolov3.txt"

    def get_output_layers(self, net):

        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers


    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, colours):

        label = str(classes[class_id])

        color = colours[class_id]

        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run_dnn(self, srcpath="path", dstpath = "path"):
        classes = None
        COLORS = None
        maxsize = 256
        minsize = 0
        filelist = os.walk(srcpath)
        for root, dir, file in filelist:
            for ii in file:
                filepath = os.path.join(root, ii)

                srcImg = cv2.imread(filepath)
                image = cv2.resize(srcImg, (maxsize, maxsize))
                #cv2.imshow("original image", image)
                height, width, shape = image.shape

                # print("file: ", ii, "dimensions h:"+ str(height) + " width: " + str(width))

                Width = image.shape[1]
                Height = image.shape[0]
                scale = 0.00392

                with open(self.argClass, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]

                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

                net = cv2.dnn.readNet(self.argWeight, self.argConfig)

                blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

                net.setInput(blob)

                outs = net.forward(self.get_output_layers(net))


                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.4

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0] * Width)
                            center_y = int(detection[1] * Height)
                            w = int(detection[2] * Width)
                            h = int(detection[3] * Height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])

                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                for i in indices:
                    j = i[0]
                    box = boxes[j]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    self.draw_prediction(image, class_ids[j], confidences[j], round(x), round(y), round(x+w), round(y+h), classes, COLORS)
                    print("file: ", ii, " x:", x, "y:", y, " dimensions h:" + str(h) + " width: " + str(w))
                    #cv2.imshow("object detection", image)

                    x1 = round(x)
                    y1 = round(y)
                    x2 = round(w) + x1
                    y2 = round(h) + y1

                    if x1 < 0 or x1 == None:
                        x1 = minsize

                    if y1 < 0 or y1 == None:
                        y1 = minsize

                    if x2 > maxsize or x2 == None:
                        x2 = maxsize

                    if y2 > maxsize or y2 == None:
                        y2 = maxsize

                    print("cropped x: " + str(x1) + " y: " + str(y1) + " w: " + str(x2) + " h: " + str(y2))


                    croppedImg = image[y1:y2, x1:x2]
                    gray = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY)
                    width, height = gray.shape
                    ratio = 100 / float(width)
                    h = int(height * ratio)
                    w = 100
                    aspect = cv2.resize(gray, (w,h), interpolation=cv2.INTER_AREA)
                    imgfile = cv2.resize(aspect, (100, 100), interpolation=cv2.INTER_AREA)
                    normalised = cv2.equalizeHist(imgfile)
                    #cv2.imshow("gray", gray)
                    outputpath = os.path.join(dstpath, ii)
                    cv2.imwrite(outputpath, normalised)

                #cv2.waitKey()

                #cv2.destroyAllWindows()

# # handle command line arguments
# arguments = argparse.ArgumentParser()
# arguments.add_argument('-i', '--image', required=True,
#                        help = 'path to input image')
# arguments.add_argument('-c', '--config', required=True,
#                        help = 'path to yolo config file')
# arguments.add_argument('-w', '--weights', required=True,
#                        help = 'path to yolo pre-trained weights')
# arguments.add_argument('-cl', '--classes', required=True,
#                        help = 'path to text file containing class names')
# args = arguments.parse_args()


