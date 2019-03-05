# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging
import tensorflow as tf
import util
from layer import (weight_variable, weight_variable_devonc, bias_variable,
                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax,
                            cross_entropy)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_conv_net(x, keep_prob, channels, layers, features_root=16, filter_size=3, pool_size=2, training = True):
    
    """
    주어진 파라미터를 이용해서 convolution u-net 그래프 생성 함 
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image
        batch_size = tf.shape(x_image)[0]

    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    in_size = 1000
    size = in_size
    
    # down layers
    for layer in range(0, layers):
        with tf.name_scope("down_conv_{}".format(str(layer))):
            features = 2 ** layer * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev, name="w1")
            else:
                w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1")

            w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
            b1 = bias_variable([features], name="b1")
            b2 = bias_variable([features], name="b2")

            conv1 = conv2d(in_node, w1, b1, keep_prob)
            tmp_h_conv = tf.nn.relu(tf.layers.batch_normalization(conv1, training = training))
            conv2 = conv2d(tmp_h_conv, w2, b2, keep_prob)
            dw_h_convs[layer] = tf.nn.relu(tf.layers.batch_normalization(conv2, training = training))

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size -= 4
            if layer < layers - 1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= 2

    in_node = dw_h_convs[layers - 1]

    # up layers
    for layer in range(layers - 2, -1, -1):
        with tf.name_scope("up_conv_{}".format(str(layer))):
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))

            wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
            bd = bias_variable([features // 2], name="bd")
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
            w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
            b1 = bias_variable([features // 2], name="b1")
            b2 = bias_variable([features // 2], name="b2")

            conv1 = conv2d(h_deconv_concat, w1, b1, keep_prob)
            h_conv = tf.nn.relu(tf.layers.batch_normalization(conv1, training=training))
            conv2 = conv2d(h_conv, w2, b2, keep_prob)
            in_node = tf.nn.relu(tf.layers.batch_normalization(conv2, training=training))
            up_h_convs[layer] = in_node
            
            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size *= 2
            size -= 4

    # Output Map
    with tf.name_scope("output_map"):
        weight = weight_variable([1, 1, features_root, 1], stddev)  # 불량 CLASS의 MAP만 Loss함수에 들어가므로 channel은 "1"이 됨.
        bias = bias_variable([1], name="bias")
        conv = conv2d(in_node, weight, bias, tf.constant(1.0))
        output_map = tf.squeeze(tf.nn.sigmoid(conv),axis=-1)
        up_h_convs["out"] = output_map                              
        
    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output_map, variables, int(in_size - size)


class Unet(object):
    """
    A unet implementation
    :param channels: (optional) number of channels in the input image
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, channels=3, cost="cross_entropy", cost_kwargs={}, **kwargs): # graph 생성

        tf.reset_default_graph()
    
        self.x = tf.placeholder(tf.float32, shape=[None, None, None, channels], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, None, None], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")   # Dropout을 위함
        self.is_training = tf.placeholder(tf.bool, name="batch_normalization") # Batch normalization을 위함

        logits, self.variables, self.offset = create_conv_net(self.x, self.keep_prob, channels,  **kwargs) 
       
        self.cost = self._get_cost(logits, cost, cost_kwargs)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        with tf.name_scope("cross_entropy"):
            self.cross_entropy = cross_entropy(tf.reshape(self.y, [-1, 1]),                      
                                               tf.reshape(logits, [-1, 1]))

        with tf.name_scope("results"):
            self.predicter = logits # logit은 각 픽셀의 sigmoid 결과 값임 (1에 가까울 수록 불량으로 예측)  
            self.predicter = tf.identity(self.predicter, name="predicter")
            self.correct_pred = tf.equal(self.predicter, self.y)   #logit: [batch, nx, ny, 1], self.y : [batch, nx, nx]
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32)) 

    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Cross_entropy, BCE_dice_coefficient, Dice_coefficient 중에서 하나의 Loss를 구해서 계산 한다.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance (Cross_entropy만 해당 함)
        regularizer: power of the L2 regularizers added to the loss function 
        """

        with tf.name_scope("cost"):
            flat_logits = tf.reshape(logits, [-1,1])
            flat_labels = tf.reshape(self.y, [-1,1])
            
            if cost_name == "cross_entropy":
                class_weights = cost_kwargs.pop("class_weights", None)

                if class_weights is not None:
                    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

                    weight_map = tf.multiply(flat_labels, class_weights)
                    weight_map = tf.reduce_sum(weight_map, axis=1)

                    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                          labels=flat_labels)
                    weighted_loss = tf.multiply(loss_map, weight_map)

                    loss = tf.reduce_mean(weighted_loss)

                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                     labels=flat_labels))
            elif cost_name == "dice_coefficient":
                smooth = 1.
                intersection = tf.reduce_sum(flat_logits * flat_labels)
                score = (2 * intersection + smooth) / (tf.reduce_sum(flat_labels) + tf.reduce_sum(flat_logits) + smooth)
                loss = 1 - score

            elif cost_name == "bce_dice_coefficient":
                smooth = 1.            
                intersection = tf.reduce_sum(flat_logits * flat_labels)
                dice_score = (2 * intersection + smooth) / (tf.reduce_sum(flat_labels) + tf.reduce_sum(flat_logits) + smooth)
                dice_loss = 1 - dice_score
                cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,labels=flat_labels))
                loss = dice_loss + cross_entropy_loss
    
            else:
                raise ValueError("Unknown cost function: " % cost_name)

            regularizer = cost_kwargs.pop("regularizer", None)
            if regularizer is not None:
                regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
                loss += (regularizer * regularizers)

            return loss


class Trainer(object):
    """
    U-net 인스턴스를 학습 한다.
    :param net: the unet instance to train
    :param data_provider: the data_provider instance to get data
    :param batch_size: size of training batch
    :param validation_batch_size: size of validation batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """
    def __init__(self, net, data_provider, batch_size=1, validation_batch_size = 4, optimizer="momentum", lr = 0.05, opt_kwargs={}):

        self.net = net                              
        self.data_provider = data_provider
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')


    def _get_optimizer(self, training_iters):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.optimizer == "momentum":
                learning_rate = self.opt_kwargs.pop("learning_rate", self.lr)
                decay_rate = self.opt_kwargs.pop("decay_rate", 0.99)
                momentum = self.opt_kwargs.pop("momentum", 0.9)

                self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                                     global_step=self.global_step,
                                                                     decay_steps=training_iters,
                                                                     decay_rate=decay_rate,
                                                                     staircase=True)

                optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                       **self.opt_kwargs).minimize(self.net.cost,
                                                                                   global_step=self.global_step)
            elif self.optimizer == "adam":
                learning_rate = self.opt_kwargs.pop("learning_rate",self.lr)
                self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node,
                                                   **self.opt_kwargs).minimize(self.net.cost,
                                                                               global_step=self.global_step)

        return optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]), name="norm_gradients")
        self.optimizer = self._get_optimizer(training_iters)
        init = tf.global_variables_initializer()

        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1,
              restore=False, write_graph=True, prediction_path='prediction'):

        """
        학습 프로세스를 수행 한다.
        :param data_provider: callable returning training and validation data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """               
        save_path = os.path.join(output_path, "model.ckpt")                                                         # epoch에 맞는 ckpt로 변경 

        if epochs == 0:
            return save_path
        
        init = self._initialize(training_iters, output_path, restore, prediction_path)
        train_init, test_init = self.data_provider.build_get_data()
        saver = tf.train.Saver(max_to_keep = 1000)  

        with tf.Session() as sess:                                                                            
            sess.run(init)
            sess.run(test_init)
            test_x, test_y = sess.run([self.data_provider.images_test, self.data_provider.gts_test])  
            pred_shape = self.store_prediction(sess, test_x, test_y, "_init")
            logging.info("Start optimization")
            avg_gradients = None
                            
            for epoch in range(epochs):
                sess.run(train_init)                                                                        
                total_loss = 0                
                try:

                    for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                        #만약에 epoch이 10, tra. iter = 30 이라면, step은 range(0, 30) 0...29 range(30, 60) 30... 59
                        #여기서 iter 수는 각 epoch 당 iter 수 이다. 즉, 각 epoch당 얼만큼의 이미지를 random으로 추출해서 학습하는가를 보는 것.
                        #따라서 epoch이 바뀔때마다 data를 다시 initializier하고, epoch당 iter x batch_size만큼 이미지를 학습한다.
                        #필요 시, 총 iter 수를 자동으로 하게 할 수도 있다. n.iter = (n.data / n.batch) * n.epoch

                        batch_x, batch_y = sess.run([self.data_provider.images, self.data_provider.gts])

                        # Run optimization op (backprop)
                        _, loss, lr, gradients = sess.run(
                            (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                            feed_dict={self.net.x: batch_x,
                                       self.net.y: batch_y,
                                       self.net.keep_prob: dropout,
                                       self.net.is_training : True })

                        if step % display_step == 0:
                            self.output_minibatch_stats(sess,batch_x,batch_y)

                        total_loss += loss
                
                except tf.errors.OutOfRangeError:
                    pass

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                test_x, test_y = sess.run([self.data_provider.images_test, self.data_provider.gts_test]) 
                self.store_prediction(sess, test_x, test_y, "epoch_%s" % epoch)
                saver.save(sess, save_path, self.global_step, write_meta_graph = True)        
            
            logging.info("Optimization Finished!")
                    
    def store_prediction(self, sess, batch_x, batch_y, name):                                                       
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x,
                                                             self.net.y: batch_y,
                                                             self.net.keep_prob: 1.})
        pred_shape = prediction.shape

        loss = sess.run(self.net.cost, feed_dict={self.net.x: batch_x,
                                                  self.net.y: batch_y,
                                                  self.net.keep_prob: 1.})

        logging.info("Validation loss={:.4f}".format(loss))

        img = util.combine_img_prediction(batch_x, batch_y, prediction)                                               
        util.save_image(img, "%s/%s.jpg" % (self.prediction_path, name))

        return pred_shape


    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, batch_x, batch_y):
        loss, correct, acc, predictions = sess.run((    self.net.cost,
                                                        self.net.correct_pred,
                                                        self.net.accuracy,
                                                        self.net.predicter),
                                                        feed_dict={self.net.x: batch_x,
                                                                  self.net.y: batch_y,
                                                                  self.net.keep_prob: 1.,
                                                                  self.net.is_training : False })

            
#         logging.info(
#             "Iter {:}, Minibatch Loss= {:.4f}, Correct {:}, Training Accuracy= {:.10f}".format(self.global_step.eval(),
#                                                                                                            loss,
#                                                                                                            correct.sum(),
#                                                                                                            acc))

        logging.info(
        "Iter {:}, Minibatch Loss= {:.4f}".format(self.global_step.eval(),loss)
