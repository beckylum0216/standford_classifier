import math
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score, precision_score



class NeuralNet(object):

    def __init__(self, w, h, outputNodes, poptrain, poptest, categories):
        # to use the correct gpu driver
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


        # Useful numbers
        self.classes = categories
        self.width = w
        self.height = h
        self.numInputNodes = w * h
        self.numOutputNodes = outputNodes
        self.populationtrain = poptrain
        self.populationtest = poptest
        self.numHiddenNode1 = 2048
        self.numHiddenNode2 = 1024
        self.numHiddenNode3 = 512
        self.numHiddenNode4 = 256
        self.numHiddenNode5 = 256
        self.numHiddenNode6 = 256


        #self.epoch = 1
        self.learnRate = 0.0001
        self.ops = {}

    def CNeuralNet(self, imageset, labelset, imagetest, labeltest, epoch, basename, batchsize):
        tf.compat.v1.disable_eager_execution()

        sess = tf.compat.v1.Session()
        # graph inputs
        x_inputs = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, self.width, self.height, 1], name='x_inputs')
        y_label = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None, self.numOutputNodes], name='y_label')


        #convolutions
        convolutionOne = tf.compat.v1.layers.conv2d(inputs=x_inputs, kernel_size = 2,filters = 2, strides = 2, padding = "VALID" )
        c1shape = convolutionOne.get_shape().as_list()
        calc1dims = 100 - 2 / 2 +1
        print("convolution 1 shape", c1shape, calc1dims)

        poolOne = tf.compat.v1.layers.max_pooling2d(convolutionOne, 2, strides = 2, padding = "VALID")
        p1shape = tf.compat.v1.shape(poolOne)
        print("pool1 shape", p1shape)


        convolutionTwo = tf.compat.v1.layers.conv2d(inputs = poolOne, kernel_size = 2, filters = 2, strides = 2, padding = "VALID")
        c2shape = tf.compat.v1.shape(convolutionTwo)
        print("convolution2 shape", c2shape)

        poolTwo = tf.compat.v1.layers.max_pooling2d(convolutionTwo, 2, strides = 2, padding = "VALID")

        p2shape = poolTwo.get_shape()
        size = p2shape[1:4].num_elements()
        print("pool2 shape", size)

        data = tf.compat.v1.reshape(poolTwo, [-1, 72])
        datashape = data.get_shape().as_list()
        print("data shape", datashape)

        # weights
        weightOne = tf.compat.v1.Variable(tf.compat.v1.random_normal([72,
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

        weightOut = tf.compat.v1.Variable(tf.compat.v1.random_normal([self.numHiddenNode6,
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
        predOut = tf.compat.v1.add(tf.compat.v1.matmul(predictionSix, weightOut), biasOut, name='predOut')
        prediction = tf.compat.v1.nn.softmax(predOut, name='prediction')

        # backpropagation
        theLogits = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=predOut, labels=y_label, name='theLogits')
        loss = tf.compat.v1.reduce_mean(theLogits, name='loss')
        optimiser = tf.compat.v1.train.AdamOptimizer(self.learnRate).minimize(loss, name='optimiser')
        prediction_final = tf.compat.v1.reshape(prediction, [-1, 196])
        correct_prediction = tf.compat.v1.equal(tf.compat.v1.argmax(prediction_final, 1),
                                      tf.compat.v1.argmax(y_label, 1), name='correct_prediction')

        accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.compat.v1.float32), name='accuracy')
        #confusion_train = tf.compat.v1.confusion_matrix(y_label, tf.compat.v1.argmax(prediction,1), num_classes = self.classes)



        # actual
        init = tf.compat.v1.global_variables_initializer()



        sess.run(init)

        lossx = 0
        accx = 0
        lossy = 0
        accy = 0
        for ii in range(epoch):
            trainbatch = self.BatchGenerator(batchsize, self.populationtrain, imageset, labelset)
            for count, (x_train, y_train) in enumerate(trainbatch):
                sess.run(optimiser, feed_dict={x_inputs: x_train, y_label: y_train})

                # checktrain = sess.run(prediction, feed_dict={x_inputs: x_train, y_label: y_train})
                # check_label_train = sess.run(y_label, feed_dict={y_label: y_train})
                #
                # predtrain= []
                # labeltrain = []
                # for ii in range(len(y_train)):
                #     predtrain.append(checktrain[ii].argmax())
                #     labeltrain.append(check_label_train[ii].argmax())
                #     print("predy: ", predtrain[ii], " labely: ", labeltrain[ii])

            if ii % 100 == 0:
                # Analyse progress so far
                lossx = sess.run(loss, feed_dict={x_inputs: x_train, y_label: y_train})
                accx = sess.run(accuracy, feed_dict={x_inputs: x_train, y_label: y_train})

                # f1 = f1_score(y_true=np.argmax(y_label,1) , y_pred=np.argmax(prediction, 1))
                # precision = precision_score(y_true=np.argmax(y_label,1) , y_pred=np.argmax(prediction, 1))
                # recall = recall_score(y_true=np.argmax(y_label,1) , y_pred=np.argmax(prediction, 1))
                #
                # print("f1: ", f1, " precision: ", precision, " recall: ", recall)


                print('Training Step:' + str(ii) + ' out of ' +
                      str(epoch) + '  Accuracy =  ' + str(accx) +
                      '  Loss = ' + str(lossx))

        # cm = sess.run(confusion_train)
        # print(cm)



        print("Training finished...")

        print("Integrated Testing")

        testbatch = self.BatchGenerator(batchsize, self.populationtest, imagetest, labeltest)
        for count, (x_test, y_test) in enumerate(testbatch):

            # actual testing
            check = sess.run(prediction, feed_dict={x_inputs: x_test, y_label: y_test})
            check_label = sess.run(y_label, feed_dict={y_label: y_test})

            predy = []
            labely = []
            for ii in range(len(y_test)):
                predy.append(check[ii].argmax())
                labely.append(check_label[ii].argmax())
                print("predy: ", predy[ii], " labely: ", labely[ii])



            lossy = sess.run(loss, feed_dict={x_inputs: x_test, y_label: y_test})
            accy = sess.run(accuracy, feed_dict={x_inputs: x_test, y_label: y_test})

            print('Accuracy = ' + str(accy) + ' Loss = ' + str(lossy))

        saver = tf.compat.v1.train.Saver()
        saver.save(sess, basename)


    def LoadCNN(self, path):
        sess = tf.compat.v1.saved_model.load(path)

    def BatchGenerator(self, batchsize, population, imgset, lblset):
        batchindex = math.ceil(population/ batchsize)

        for ii in range(batchindex):
            if ii == batchindex-1:
                lblbatch = lblset[-batchsize:]
                imgbatch = imgset[-batchsize:]
                yield  (imgbatch, lblbatch)
            else:
                lblbatch = lblset[batchsize * ii : batchsize * ii + batchsize]
                imgbatch = imgset[batchsize * ii : batchsize * ii + batchsize]
                yield (imgbatch, lblbatch)
