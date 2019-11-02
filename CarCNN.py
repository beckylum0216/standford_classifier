import math
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score, precision_score
import sklearn
import matplotlib.pyplot as plt
import seaborn
import Utility

class NeuralNet(object):

    def __init__(self, w, h, outputNodes, poptrain, poptest):
        # to use the correct gpu driver
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


        # Useful numbers
        self.classes = outputNodes
        print("classes", self.classes)
        self.width = w
        self.height = h
        self.numInputNodes = w * h
        self.numOutputNodes = outputNodes
        print("outputnodes", self.numOutputNodes)
        self.populationtrain = poptrain
        self.populationtest = poptest
        self.numHiddenNode1 = 2048
        self.numHiddenNode2 = 1024
        self.numHiddenNode3 = 512
        self.numHiddenNode4 = 256
        self.numHiddenNode5 = 256
        self.numHiddenNode6 = 196

        self.learnRate = 0.0001
        self.dropoutRate = 0.5
        self.ops = {}
        self.annopath = "car_labels/names.txt"

    def TrainCNN(self, imageset, labelset, imagetest, labeltest, epoch, basename, origin, batchsize, map):
        tf.compat.v1.disable_eager_execution()

        sess = tf.compat.v1.Session()
        # graph inputs
        x_inputs = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, self.width, self.height, 1], name='x_inputs')
        y_label = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, self.numOutputNodes], name='y_label')

        #convolutions
        convolutionOne = tf.compat.v1.layers.conv2d(inputs=x_inputs, kernel_size = 2,filters = 2, strides = 2, padding = "VALID" )
        c1shape = convolutionOne.get_shape().as_list()
        # print("convolution 1 shape", c1shape)

        poolOne = tf.compat.v1.layers.max_pooling2d(convolutionOne, 2, strides = 2, padding = "VALID")
        p1shape = poolOne.get_shape()
        p1size = p1shape[1:4].num_elements()
        # print("pool1 shape", p1size)


        convolutionTwo = tf.compat.v1.layers.conv2d(inputs = poolOne, kernel_size = 2, filters = 2, strides = 2, padding = "VALID")
        c2shape = tf.compat.v1.shape(convolutionTwo)
        # print("convolution2 shape", c2shape)

        poolTwo = tf.compat.v1.layers.max_pooling2d(convolutionTwo, 2, strides = 2, padding = "VALID")

        p2shape = poolTwo.get_shape()
        p2size = p2shape[1:4].num_elements()
        # print("pool2 shape", p2size)

        data = tf.compat.v1.reshape(poolOne, [-1, 1250])
        datashape = data.get_shape().as_list()
        # print("data shape", datashape)

        # weights
        weightOne = tf.compat.v1.Variable(tf.compat.v1.random_normal([1250,
                                                  self.numHiddenNode1]), name='weightOne')

        weightTwo = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode1,
                                                  self.numHiddenNode2]), name='weightTwo')

        weightThree = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode2,
                                                  self.numHiddenNode3]), name='weightThree')

        weightFour = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode3,
                                                   self.numHiddenNode4]), name='weightFour')

        weightFive = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode4,
                                                   self.numHiddenNode5]), name='weightFive')

        weightSix = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode5,
                                                   self.numHiddenNode6]), name='weightSix')

        weightOut = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode5,
                                                  self.numOutputNodes]), name='weightOut')
        # biases
        biasOne = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode1]), name='biasOne')
        biasTwo = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode2]), name='biasTwo')
        biasThree = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode3]), name='biasThree')
        biasFour = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode4]), name='biasFour')
        biasFive = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode5]), name='biasFive')
        biasSix = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode6]), name='biasSix')
        biasOut = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numOutputNodes]), name='biasOut')

        # forward propagation
        predictionOne = tf.compat.v1.add(tf.compat.v1.matmul(data, weightOne), biasOne, name='predictionOne')

        predictionTwo = tf.compat.v1.add(tf.compat.v1.matmul(predictionOne, weightTwo), biasTwo, name='predictionTwo')
        predictionThree = tf.compat.v1.add(tf.compat.v1.matmul(predictionTwo, weightThree), biasThree, name='predictionThree')
        predictionFour = tf.compat.v1.add(tf.compat.v1.matmul(predictionThree, weightFour), biasFour, name='predictionFour')
        predictionFive = tf.compat.v1.add(tf.compat.v1.matmul(predictionFour, weightFive), biasFive, name='predictionFive')
        predictionSix = tf.compat.v1.add(tf.compat.v1.matmul(predictionFive, weightSix), biasSix, name='predictionSix')
        predOut = tf.compat.v1.add(tf.compat.v1.matmul(predictionFive, weightOut), biasOut, name='predOut')
        prediction = tf.compat.v1.nn.softmax(predOut, name='prediction')

        # dropoutOne = tf.compat.v1.nn.dropout(predictionOne, keep_prob=self.dropoutRate)

        # backpropagation
        theLogits = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=predOut, labels=y_label, name='theLogits')
        loss = tf.compat.v1.reduce_mean(theLogits, name='loss')
        optimiser = tf.compat.v1.train.AdamOptimizer(self.learnRate).minimize(loss, name='optimiser')
        prediction_final = tf.compat.v1.reshape(prediction, [-1, self.classes])
        correct_prediction = tf.compat.v1.equal(tf.compat.v1.argmax(prediction_final, 1),
                                      tf.compat.v1.argmax(y_label, 1), name='correct_prediction')

        accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.compat.v1.float32), name='accuracy')

        # actual
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        lossx = 0
        accx = 0
        lossy = 0
        accy = 0
        trainbatch = self.BatchGenerator(batchsize, self.populationtrain, imageset, labelset)
        # print("batch", len(trainbatch))
        for ii in range(epoch):
            count = 0
            for (x_train, y_train) in trainbatch:
                # print("count", count)
                count += 1
                #print("trainign batch",y_train)
                sess.run(optimiser, feed_dict={x_inputs: x_train, y_label: y_train})

                # checktrain = sess.run(prediction, feed_dict={x_inputs: x_train, y_label: y_train}).argmax()
                # check_label_train = sess.run(y_label, feed_dict={y_label: y_train}).argmax()
                # print("predx: ", checktrain, " labelx: ", check_label_train)

                #
                # predtrain= []
                # labeltrain = []
                # for ii in range(len(y_train)):
                #     predtrain.append(checktrain[ii].argmax())
                #     labeltrain.append(check_label_train[ii].argmax())
                #     print("predy: ", predtrain[ii], " labely: ", labeltrain[ii])

            x_train = imageset
            y_train = labelset
            if ii % 100 == 0:
                # Analyse progress so far
                lossx = sess.run(loss, feed_dict={x_inputs: x_train, y_label:y_train})
                accx = sess.run(accuracy, feed_dict={x_inputs: x_train, y_label: y_train})

                print('Training Step:' + str(ii) + ' out of ' +
                      str(epoch) + '  Accuracy =  ' + str(accx) +
                      '  Loss = ' + str(lossx))


        print("Training finished...")

        print("Integrated Testing")

        predy = []
        labely = []
        testbatch = self.BatchGenerator(batchsize, self.populationtest, imagetest, labeltest)
        index = 0
        for (x_test, y_test) in testbatch:

            # actual testing
            check = sess.run(prediction, feed_dict={x_inputs: x_test, y_label: y_test})
            check_label = sess.run(y_label, feed_dict={y_label: y_test})

            # checktrain = sess.run(prediction, feed_dict={x_inputs: x_test, y_label: y_test}).argmax()
            # check_label_train = sess.run(y_label, feed_dict={y_label: y_test}).argmax()
            # print("predy: ", checktrain, " labely: ", check_label_train)

            for ii in range(len(y_test)):
                predy.append(check[ii].argmax())
                labely.append(check_label[ii].argmax())
                index += 1



        x_test = imagetest
        y_test = labeltest
        ut = Utility.Utility()
        lossy = sess.run(loss, feed_dict={x_inputs: x_test, y_label: y_test})
        accy = sess.run(accuracy, feed_dict={x_inputs: x_test, y_label: y_test})
        print('Accuracy = ' + str(accy) + ' Loss = ' + str(lossy))


        for ii in range(index):
            print("predy: ", predy[ii], " labely: ", labely[ii])
            # ut.ShowResult(origin[ii], self.annopath, predy[ii], map)

        confusiontrain = sklearn.metrics.confusion_matrix(labely, predy)
        classnames = list(range(self.classes))
        sn = seaborn.heatmap(confusiontrain, annot=True, annot_kws={"size": 8}, cbar=False)
        plt.show()
        pltpath = "plots/confusion.png"
        sn.figure.savefig(pltpath)

        cr = sklearn.metrics.classification_report(labely, predy, labels=classnames)
        print(cr)

        histo = ut.PredictionHisto(predy, self.classes)

        plt.hist(histo, self.classes)
        histopath = "plots/histo.png"
        plt.savefig(histopath)



        saver = tf.compat.v1.train.Saver()
        saver.save(sess, basename)


    def LoadCNN(self, path, popsize, targetImgs, targetLbls, origin, batchsize, map):
        # print("pop size; ", popsize)
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()

        sess = tf.compat.v1.Session()
        # graph inputs
        x_inputs = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, self.width, self.height, 1],
                                            name='x_inputs')
        y_label = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, self.numOutputNodes], name='y_label')

        # convolutions
        convolutionOne = tf.compat.v1.layers.conv2d(inputs=x_inputs, kernel_size=2, filters=2, strides=2,
                                                    padding="VALID")
        c1shape = convolutionOne.get_shape().as_list()
        #print("convolution 1 shape", c1shape)

        poolOne = tf.compat.v1.layers.max_pooling2d(convolutionOne, 2, strides=2, padding="VALID")
        p1shape = poolOne.get_shape()
        p1size = p1shape[1:4].num_elements()
        #print("pool1 shape", p1size)

        convolutionTwo = tf.compat.v1.layers.conv2d(inputs=poolOne, kernel_size=2, filters=2, strides=2,
                                                    padding="VALID")
        c2shape = tf.compat.v1.shape(convolutionTwo)
        #print("convolution2 shape", c2shape)

        poolTwo = tf.compat.v1.layers.max_pooling2d(convolutionTwo, 2, strides=2, padding="VALID")

        p2shape = poolTwo.get_shape()
        p2size = p2shape[1:4].num_elements()
        #print("pool2 shape", p2size)

        data = tf.compat.v1.reshape(poolOne, [-1, 1250])
        datashape = data.get_shape().as_list()
        #print("data shape", datashape)

        # weights
        weightOne = tf.compat.v1.Variable(tf.compat.v1.random_normal([1250,
                                                                      self.numHiddenNode1]), name='weightOne')

        weightTwo = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode1,
                                                                      self.numHiddenNode2]), name='weightTwo')

        weightThree = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode2,
                                                                        self.numHiddenNode3]), name='weightThree')

        weightFour = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode3,
                                                                       self.numHiddenNode4]), name='weightFour')

        weightFive = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode4,
                                                                       self.numHiddenNode5]), name='weightFive')

        weightSix = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode5,
                                                                      self.numHiddenNode6]), name='weightSix')

        weightOut = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode5,
                                                                      self.numOutputNodes]), name='weightOut')
        # biases
        biasOne = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode1]), name='biasOne')
        biasTwo = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode2]), name='biasTwo')
        biasThree = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode3]), name='biasThree')
        biasFour = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode4]), name='biasFour')
        biasFive = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode5]), name='biasFive')
        biasSix = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode6]), name='biasSix')
        biasOut = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numOutputNodes]), name='biasOut')

        # forward propagation
        predictionOne = tf.compat.v1.add(tf.compat.v1.matmul(data, weightOne), biasOne, name='predictionOne')
        predictionTwo = tf.compat.v1.add(tf.compat.v1.matmul(predictionOne, weightTwo), biasTwo, name='predictionTwo')
        predictionThree = tf.compat.v1.add(tf.compat.v1.matmul(predictionTwo, weightThree), biasThree,
                                           name='predictionThree')
        predictionFour = tf.compat.v1.add(tf.compat.v1.matmul(predictionThree, weightFour), biasFour,
                                          name='predictionFour')
        predictionFive = tf.compat.v1.add(tf.compat.v1.matmul(predictionFour, weightFive), biasFive,
                                          name='predictionFive')
        predictionSix = tf.compat.v1.add(tf.compat.v1.matmul(predictionFive, weightSix), biasSix, name='predictionSix')
        predOut = tf.compat.v1.add(tf.compat.v1.matmul(predictionFive, weightOut), biasOut, name='predOut')
        prediction = tf.compat.v1.nn.softmax(predOut, name='prediction')

        # backpropagation
        theLogits = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=predOut, labels=y_label, name='theLogits')
        loss = tf.compat.v1.reduce_mean(theLogits, name='loss')
        optimiser = tf.compat.v1.train.AdamOptimizer(self.learnRate).minimize(loss, name='optimiser')
        prediction_final = tf.compat.v1.reshape(prediction, [-1, self.classes])
        correct_prediction = tf.compat.v1.equal(tf.compat.v1.argmax(prediction_final, 1),
                                                tf.compat.v1.argmax(y_label, 1), name='correct_prediction')

        accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.compat.v1.float32),
                                            name='accuracy')



        # actual
        init = tf.compat.v1.global_variables_initializer()

        sess.run(init)

        saver = tf.compat.v1.train.Saver()

        saver.restore(sess, path)
        testbatch = self.BatchGenerator(batchsize, popsize, targetImgs, targetLbls)

        count = 0
        predy = []
        labely = []
        ut = Utility.Utility()
        for (x_test, y_test) in testbatch:

            # actual testing
            check = sess.run(prediction, feed_dict={x_inputs: x_test, y_label: y_test})
            check_label = sess.run(y_label, feed_dict={y_label: y_test})

            # checktest = sess.run(prediction, feed_dict={x_inputs: x_test, y_label: y_test}).argmax()
            # check_label_test = sess.run(y_label, feed_dict={y_label: y_test}).argmax()
            # print("pred test: ", checktest, " label test: ", check_label_test)


            for ii in range(len(y_test)):
                predy.append(check[ii].argmax())
                labely.append(check_label[ii].argmax())
                count += 1

        for ii in range(count):
            print("predy list: ", predy[ii], " labely list: ", labely[ii])
            # ut.ShowResult(origin[ii], self.annopath, predy[ii], map)

        x_test = targetImgs
        y_test = targetLbls

        lossy = sess.run(loss, feed_dict={x_inputs: x_test, y_label: y_test})
        accy = sess.run(accuracy, feed_dict={x_inputs: x_test, y_label: y_test})
        print('Accuracy = ' + str(accy) + ' Loss = ' + str(lossy))


        confusiontrain = sklearn.metrics.confusion_matrix(labely, predy)
        classnames = list(range(self.classes))
        sn = seaborn.heatmap(confusiontrain, annot=True, annot_kws={"size": 3}, cbar=False)
        plt.show()
        pltpath = "plots/confusion.png"
        sn.figure.savefig(pltpath)

        cr = sklearn.metrics.classification_report(labely, predy, labels=classnames)
        print(cr)


        histo = ut.PredictionHisto(predy, self.classes)
        plt.hist(histo, classnames)
        histopath = "plots/histo.png"
        plt.savefig(histopath)



    def PredictCNN(self, path, popsize, targetImgs, targetLbls, origin, batchsize, map):
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()

        sess = tf.compat.v1.Session()
        # graph inputs
        x_inputs = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, self.width, self.height, 1],
                                            name='x_inputs')
        y_label = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, self.numOutputNodes], name='y_label')

        # convolutions
        convolutionOne = tf.compat.v1.layers.conv2d(inputs=x_inputs, kernel_size=2, filters=2, strides=2,
                                                    padding="VALID")
        c1shape = convolutionOne.get_shape().as_list()
        # print("convolution 1 shape", c1shape)

        poolOne = tf.compat.v1.layers.max_pooling2d(convolutionOne, 2, strides=2, padding="VALID")
        p1shape = poolOne.get_shape()
        p1size = p1shape[1:4].num_elements()
        # print("pool1 shape", p1size)

        convolutionTwo = tf.compat.v1.layers.conv2d(inputs=poolOne, kernel_size=2, filters=2, strides=2,
                                                    padding="VALID")
        c2shape = tf.compat.v1.shape(convolutionTwo)
        # print("convolution2 shape", c2shape)

        poolTwo = tf.compat.v1.layers.max_pooling2d(convolutionTwo, 2, strides=2, padding="VALID")

        p2shape = poolTwo.get_shape()
        p2size = p2shape[1:4].num_elements()
        # print("pool2 shape", p2size)

        data = tf.compat.v1.reshape(poolOne, [-1, 1250])
        datashape = data.get_shape().as_list()
        # print("data shape", datashape)

        # weights
        weightOne = tf.compat.v1.Variable(tf.compat.v1.random_normal([1250,
                                                                      self.numHiddenNode1]), name='weightOne')

        weightTwo = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode1,
                                                                      self.numHiddenNode2]), name='weightTwo')

        weightThree = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode2,
                                                                        self.numHiddenNode3]), name='weightThree')

        weightFour = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode3,
                                                                       self.numHiddenNode4]), name='weightFour')

        weightFive = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode4,
                                                                       self.numHiddenNode5]), name='weightFive')

        weightSix = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode5,
                                                                      self.numHiddenNode6]), name='weightSix')

        weightOut = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode5,
                                                                      self.numOutputNodes]), name='weightOut')
        # biases
        biasOne = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode1]), name='biasOne')
        biasTwo = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode2]), name='biasTwo')
        biasThree = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode3]), name='biasThree')
        biasFour = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode4]), name='biasFour')
        biasFive = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode5]), name='biasFive')
        biasSix = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode6]), name='biasSix')
        biasOut = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numOutputNodes]), name='biasOut')

        # forward propagation
        predictionOne = tf.compat.v1.add(tf.compat.v1.matmul(data, weightOne), biasOne, name='predictionOne')
        predictionTwo = tf.compat.v1.add(tf.compat.v1.matmul(predictionOne, weightTwo), biasTwo, name='predictionTwo')
        predictionThree = tf.compat.v1.add(tf.compat.v1.matmul(predictionTwo, weightThree), biasThree,
                                           name='predictionThree')
        predictionFour = tf.compat.v1.add(tf.compat.v1.matmul(predictionThree, weightFour), biasFour,
                                          name='predictionFour')
        predictionFive = tf.compat.v1.add(tf.compat.v1.matmul(predictionFour, weightFive), biasFive,
                                          name='predictionFive')
        predictionSix = tf.compat.v1.add(tf.compat.v1.matmul(predictionFive, weightSix), biasSix, name='predictionSix')
        predOut = tf.compat.v1.add(tf.compat.v1.matmul(predictionFive, weightOut), biasOut, name='predOut')
        prediction = tf.compat.v1.nn.softmax(predOut, name='prediction')

        # backpropagation
        theLogits = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=predOut, labels=y_label, name='theLogits')
        loss = tf.compat.v1.reduce_mean(theLogits, name='loss')
        optimiser = tf.compat.v1.train.AdamOptimizer(self.learnRate).minimize(loss, name='optimiser')
        prediction_final = tf.compat.v1.reshape(prediction, [-1, self.classes])
        correct_prediction = tf.compat.v1.equal(tf.compat.v1.argmax(prediction_final, 1),
                                                tf.compat.v1.argmax(y_label, 1), name='correct_prediction')

        accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.compat.v1.float32),
                                            name='accuracy')

        # actual
        init = tf.compat.v1.global_variables_initializer()

        sess.run(init)

        saver = tf.compat.v1.train.Saver()

        saver.restore(sess, path)
        # testbatch = self.BatchGenerator(batchsize, popsize, targetImgs, targetLbls)

        count = 0
        predy = 0
        labely = 0
        # for (x_test, y_test) in testbatch:
        #
            # actual testing
            # check = sess.run(prediction, feed_dict={x_inputs: x_test, y_label: y_test})
            # check_label = sess.run(y_label, feed_dict={y_label: y_test})
        #
        #     checktest = sess.run(prediction, feed_dict={x_inputs: x_test, y_label: y_test}).argmax()
        #     check_label_test = sess.run(y_label, feed_dict={y_label: y_test}).argmax()
        #     print("prediction: ", checktest, " label: ", check_label_test)

        #     w, h = check_label.shape
        #     # print("check count", count)
        #     for ii in range(w):
        #         predy.append(check[ii].argmax())
        #         labely.append(check_label[ii].argmax())
        #         count += 1
        #
        # for ii in range(count):
        #     print("predy list: ", predy[ii], " labely list: ", labely[ii])

        x_test = targetImgs
        y_test = targetLbls

        ut = Utility.Utility()
        checktest = sess.run(prediction, feed_dict={x_inputs: x_test, y_label: y_test}).argmax()
        checkconfidence = np.amax(sess.run(prediction, feed_dict={x_inputs: x_test, y_label: y_test}))
        check_label_test = sess.run(y_label, feed_dict={y_label: y_test}).argmax()
        print("prediction: ", checktest, " label: ", check_label_test," confidence: ", checkconfidence)
        ut.ShowResult(origin[0], self.annopath, check_label_test, map)



        # lossy = sess.run(loss, feed_dict={x_inputs: x_test, y_label: y_test})
        # accy = sess.run(accuracy, feed_dict={x_inputs: x_test, y_label: y_test})
        # print('Accuracy = ' + str(accy) + ' Loss = ' + str(lossy))

    def BatchGenerator(self, batchsize, population, imgset, lblset):
        batch = []
        # print("population", population)
        batchindex = math.ceil(population / batchsize)

        for ii in range(batchindex):
            #print("index ", ii)
            if ii == batchindex:
                lblbatch = lblset[-batchsize:]
                imgbatch = imgset[-batchsize:]
                batch.append((imgbatch, lblbatch))
            else:
                lblbatch = lblset[batchsize * ii : batchsize * ii + batchsize]
                imgbatch = imgset[batchsize * ii : batchsize * ii + batchsize]
                batch.append((imgbatch, lblbatch))

        return batch
