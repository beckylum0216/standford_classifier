# import mxnet as mx
# import numpy as np
# from mxnet import nd, autograd, gluon
# #from mxnet.ndarray import relu
#
#
# class NeuralNet(object):
#
#     def __init__(self, testImgs, testLbls, trainImgs, trainLbls, epochinput, savepath):
#         # some constants
#
#         self.epoch = epochinput
#         self.learning_rate = .01
#         self.smoothing_constant = .01
#         self.numOfInputs = 100 * 100
#         self.numOfOutputs = 196
#         self.ctx = mx.gpu()
#
#         trainImgArray = self.ListToArray(trainImgs)
#         trainLblArray = np.array(trainLbls, dtype=int)
#         testImgArray = np.array(testImgs, dtype=int)
#         testLblArray = np.array(testLbls, dtype=int)
#
#         self.ndTrainX = mx.nd.array(trainImgs)
#         print(self.ndTrainX.shape)
#         # ndTrainX.flatten()
#         self.ndTrainY = mx.nd.array(trainLblArray)
#         # ndTrainY.flatten()
#         self.ndTestX = mx.nd.array(testImgArray)
#         self.ndTestY = mx.nd.array(testLblArray)
#         self.gTrainSet = mx.io.NDArrayIter(self.ndTrainX, self.ndTrainY)
#         self.gTestSet = mx.io.NDArrayIter(self.ndTestX, self.ndTestY)
#
#         # input layer
#         self.trainData = gluon.data.DataLoader(self.gTrainSet, batch_size=10)
#         self.testData = gluon.data.DataLoader(self.gTestSet, batch_size=10)
#         # checking only
#         for ii, (data, label) in enumerate(self.trainData):
#             print((data, label))
#
#         # defining the model
#         self.weight_scale = .01
#         self.num_fc = 100
#         self.num_filter_conv_layer1 = 20
#         self.num_filter_conv_layer2 = 50
#
#         self.W1 = nd.random_normal(shape=(self.num_filter_conv_layer1, 1, 3, 3), scale=self.weight_scale, ctx=self.ctx)
#         self.b1 = nd.random_normal(shape=self.num_filter_conv_layer1, scale=self.weight_scale, ctx=self.ctx)
#
#         self.W2 = nd.random_normal(shape=(self.num_filter_conv_layer2, self.num_filter_conv_layer1, 5, 5),
#                               scale=self.weight_scale, ctx=self.ctx)
#         self.b2 = nd.random_normal(shape=self.num_filter_conv_layer2, scale=self.weight_scale, ctx=self.ctx)
#
#         self.W3 = nd.random_normal(shape=(800, self.num_fc), scale=self.weight_scale, ctx=self.ctx)
#         self.b3 = nd.random_normal(shape=self.num_fc, scale=self.weight_scale, ctx=self.ctx)
#
#         self.W4 = nd.random_normal(shape=(self.num_fc, self.numOfOutputs), scale=self.weight_scale, ctx=self.ctx)
#         self.b4 = nd.random_normal(shape=self.numOfOutputs, scale=self.weight_scale, ctx=self.ctx)
#
#         self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
#
#         for ii in self.params:
#             ii.attach_grad()
#
#     def ListToArray(self, target):
#         a = nd.empty(1, ctx=self.ctx)
#         out = nd.empty(1, ctx=self.ctx)
#         for img in target:
#             b = nd.array(img, ctx=self.ctx)
#             out = nd.concat(a, b)
#         print(out.shape)
#         return a
#
#     def Model(self, inputX, debug = False):
#         h1_conv = nd.Convolution(data=inputX, weight=self.W1, bias=self.b1, kernel=(3, 3),
#                                  num_filter=self.num_filter_conv_layer1)
#         h1_activation = self.relu(h1_conv)
#         h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2, 2), stride=(2, 2))
#         if debug:
#             print("h1 shape: %s" % (np.array(h1.shape)))
#
#         h2_conv = nd.Convolution(data=h1, weight=self.W2, bias=self.b2, kernel=(5, 5),
#                                  num_filter=self.num_filter_conv_layer2)
#         h2_activation = self.relu(h2_conv)
#         h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2, 2), stride=(2, 2))
#         if debug:
#             print("h2 shape: %s" % (np.array(h2.shape)))
#
#
#         h2 = nd.flatten(h2)
#         if debug:
#             print("Flat h2 shape: %s" % (np.array(h2.shape)))
#
#
#         h3_linear = nd.dot(h2, self.W3) + self.b3
#         h3 = self.relu(h3_linear)
#         if debug:
#             print("h3 shape: %s" % (np.array(h3.shape)))
#
#         yhat_linear = nd.dot(h3, self.W4) + self.b4
#         if debug:
#             print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))
#
#         return yhat_linear
#
#     def relu(self, X):
#         return nd.maximum(X, nd.zeros_like(X))
#
#     def softmax_cross_entropy(self, yhat_linear, y):
#         return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)
#
#     def SGD(self, params, lr):
#         for param in params:
#             param[:] = param - lr * param.grad
#
#     def evaluate_accuracy(self, data_iterator, net):
#         numerator = 0.
#         denominator = 0.
#         for i, (data, label) in enumerate(data_iterator):
#             data = data.as_in_context(self.ctx)
#             label = label.as_in_context(self.ctx)
#             label_one_hot = nd.one_hot(label, 10)
#             output = net(data)
#             predictions = nd.argmax(output, axis=1)
#             numerator += nd.sum(predictions == label)
#             denominator += data.shape[0]
#         return (numerator / denominator).asscalar()
#
#
#     def RunCNN(self):
#         moving_loss = 0
#         for e in range(self.epoch):
#             for i, (data, label) in enumerate(self.trainData):
#                 data = data.as_in_context(self.ctx)
#                 label = label.as_in_context(self.ctx)
#                 label_one_hot = nd.one_hot(label, self.numOfOutputs)
#                 with autograd.record():
#                     output = self.Model(data)
#                     loss = self.softmax_cross_entropy(output, label_one_hot)
#                 loss.backward()
#                 self.SGD(self.params, self.learning_rate)
#
#
#                 curr_loss = nd.mean(loss).asscalar()
#                 moving_loss = (curr_loss if ((i == 0) and (e == 0))
#                                else (1 - self.smoothing_constant) * moving_loss + (self.smoothing_constant) * curr_loss)
#
#             test_accuracy = self.evaluate_accuracy(self.testData, self.Model)
#             train_accuracy = self.evaluate_accuracy(self.trainData, self.Model)
#             print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
