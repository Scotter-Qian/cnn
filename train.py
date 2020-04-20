import time
import os
import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Read the training data from the file of tf.record
def read_example(filename, batch_size):
    reader = tf.TFRecordReader()

    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    _, serialized_example = reader.read(filename_queue)
    min_queue_examples = 50
    batch = tf.train.shuffle_batch([serialized_example],
                                   batch_size=batch_size,
                                   capacity=min_queue_examples + 100 * batch_size,
                                   min_after_dequeue=min_queue_examples,
                                   num_threads=2)
    parsed_example = tf.parse_example(batch, features={'image': tf.FixedLenFeature([], tf.string),
                                                       'label': tf.FixedLenFeature([], tf.int64)})
    image_raw = tf.decode_raw(parsed_example['image'], tf.uint8)
    # IMAGE_HEIGHT equals to 288，IMAGE_WIDTH equals to 384, IMAGE_DEPTH equals to 1
    image = tf.cast(tf.reshape(image_raw, [batch_size, 288, 384, 1]), tf.float32)
    image = image/255.0
    #label_raw = tf.decode_raw(parsed_example['label'], tf.int8)
    label_raw = tf.cast(parsed_example['label'], tf.int32)
    # Convert lable to one-hot form
    label = tf.reshape(label_raw, [batch_size*1])
    label = tf.one_hot(label, depth=num_classes)
    return image, label

#-----------------------------------------------------------------------------------------------------------------------
# Define the parameters for VGG net
def lrelu(input):
    return tf.nn.leaky_relu(input, name='leaky_relu')

def relu(input):
    return tf.nn.relu(input, name="relu")

def max_pooling(input, size, stride, name):
    with tf.variable_scope(name):
        x = tf.layers.max_pooling2d(input, size, stride, padding="SAME", name=name)
    return x

def average_pooling(input, size, strides, name):
    with tf.variable_scope(name):
        x = tf.layers.average_pooling2d(input, size, strides, padding="SAME", name=name)
    return x

def batch_normalization(input, training):
    shape = input.get_shape().as_list()
    dimension = shape[-1]
    if len(shape) == 4:
        mean, variance = tf.nn.moments(input, axes=[0,1,2])
    else:
        mean, variance = tf.nn.moments(input, axes=[0])
    beta = tf.get_variable("beta", dimension,tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=training)
    gamma = tf.get_variable("gamma", dimension, tf.float32, initializer=tf.constant_initializer(1.0,tf.float32), trainable=training)
    bn = tf.nn.batch_normalization(input, mean, variance, beta, gamma, variance_epsilon=0.001)
    return bn

def conv_layer(input, kernel_size, out_channels, strides, name, batch_normalize, activation, training=True):
    in_channels = input.get_shape()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable(name="weights", shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
        b = tf.get_variable(name="biases", shape=[out_channels],
                            dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=training)
        x = tf.nn.conv2d(input, w, strides, padding="SAME", name="conv")
        x = tf.nn.bias_add(x, b, name="bias_add")
        # If there is a BN layer, the BN layer should be placed between tf.layers.conv2d and activation
        if batch_normalize:
            y = batch_normalization(x, training)
            if activation:
                return activation(y)
            else:
                return y
        else:
            if activation:
                return activation(x)
            else:
                return x


def fc_layer(input, in_size, out_size, name, batch_normalize, activation, training=True):
    with tf.variable_scope(name):
        weights = tf.get_variable('fc_weights', shape=[in_size, out_size],
                                initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
        biases = tf.get_variable('fc_bias', shape=[out_size],
                                initializer=tf.constant_initializer(0.0), trainable=training)
        fc =  tf.matmul(input, weights) + biases
        if batch_normalize:
            fc = batch_normalization(fc, training)
        if activation:
            fc = activation(fc)

        return fc
#-----------------------------------------------------------------------------------------------------------------------
# define the inception module for the GoogLeNet
def inception_module(input, filters_1, filters_2_reduce, filters_2, filters_3_reduce, filters_3, filters_4, name):
    with tf.variable_scope(name):
        conv_1 = conv_layer(input=input, kernel_size=(1, 1), out_channels=filters_1, strides=[1, 1, 1, 1],
                            name="in_m1", batch_normalize=False, activation=relu, training=True)
        conv_2_reduce = conv_layer(input, (1, 1), filters_2_reduce, [1, 1, 1, 1], name="in_m2_reduce",
                            batch_normalize=False, activation=relu, training=True)
        conv_2 = conv_layer(conv_2_reduce, (3, 3), filters_2, [1, 1, 1, 1], name="in_m2",
                            batch_normalize=False, activation=relu, training=True)
        conv_3_reduce = conv_layer(input, (1, 1), filters_3_reduce, [1, 1, 1, 1], name="in_m3_reduce",
                            batch_normalize=False, activation=relu, training=True)
        conv_3 = conv_layer(conv_3_reduce, (5, 5), filters_3, [1, 1, 1, 1], name="in_m3",
                            batch_normalize=False, activation=relu, training=True)
        maxpool = max_pooling(input, (3, 3), (1, 1), name="max_pooling")
        conv_4 = conv_layer(maxpool, (1, 1), filters_4, [1, 1, 1, 1], name="in_m4",
                            batch_normalize=False, activation=relu, training=True)
        output = tf.concat([conv_1, conv_2, conv_3, conv_4], axis=3)

        return output


#-----------------------------------------------------------------------------------------------------------------------
# define the channel attention and spatial attention modules for CBAM
def cbam_block(input_feature, name, ratio=8):
    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, "ch_at", ratio)
        attention_feature = spatial_attention(attention_feature, "sp_at")
        print("CBAM Hello")
    return attention_feature

def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        print(channel)
        avg_pool = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)
        assert avg_pool.get_shape()[1:]==(1,1,channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel//ratio,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name="mlp_0",
                                   reuse=None)
        assert avg_pool.get_shape()[1:]==(1,1,channel//ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name="mlp_1",
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1,2], keepdims=True)
        assert max_pool.get_shape()[1:] == (1,1,channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel//ratio,
                                   activation=tf.nn.relu,
                                   name="mlp_0",
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1,1,channel//ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name="mlp_1",
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, "sigmoid")
        return input_feature * scale

def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1,1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name="conv")
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, "sigmoid")

        return input_feature * concat\

#-----------------------------------------------------------------------------------------------------------------------
# define the inception module for ResNet 
def building_block(inputs, filter_size, filters, strides, name):
    with tf.variable_scope(name):
        x = cbam_block(inputs,name="resNet_cbam")
        if strides[1] > 1:
            x = conv_layer(x, (1, 1), filters, strides, name="conv_x", batch_normalize=True, activation=False, training=True)
        f_x = conv_layer(inputs, filter_size, filters, strides=strides, name="conv_1", batch_normalize=True, activation=relu, training=True)
        f_x = conv_layer(f_x, filter_size, filters, strides=[1, 1, 1, 1], name="conv_2", batch_normalize=True, activation=False, training=True)
        f_x = tf.add(f_x, x)
        output = tf.nn.relu(f_x)

        return output


def calculate_accuracy(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct)*100.0

    return accuracy

def draw_loss(data):
    train_loss_array = np.array(data)
    plt.figure(figsize=[8,6])
    plt.plot(train_loss_array[:,0], train_loss_array[:,1], 'b', linewidth=2.0)
    plt.title('Train Loss Curve', fontsize=18)
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.show()

def draw_learning_rate(data):
    train_loss_array = np.array(data)
    plt.figure(figsize=[8,6])
    plt.plot(train_loss_array[:,0], train_loss_array[:,1], 'b', linewidth=2.0)
    plt.title('Learning Rate Curve', fontsize=18)
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('Learning Rate', fontsize=16)
    plt.show()


def count_numbers(self, data_path):
    num = 0
    for name in os.listdir(data_path):
        if name.endswith(".xy"):
            num += 1

    return num


#-----------------------------------------------------------------------------------------------------------------------
# Construct Models
class SelectModel(object):
    def __init__(self, num_classes, learning_rate_base, save_rate, epoch_num, batch_size, model_path):
        self.num_classes = num_classes
        self.learning_rate_base = learning_rate_base
        self.save_rate = save_rate
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.model_path = model_path

    def vggNet(self, input):
        with tf.name_scope("VGG16"):
            output_1 = conv_layer(input, (3, 3), 32, [1, 1, 1, 1], name="conv_1", batch_normalize=True, activation=lrelu)
            output_pool_1 = max_pooling(output_1, (2, 2), (2, 2), name="max_pooling_1")
            print(output_pool_1.get_shape())
            output_2 = conv_layer(output_pool_1, (3, 3), 64, [1, 1, 1, 1], name="conv_2", batch_normalize=True,
                                  activation=lrelu)
            output_pool_2 = max_pooling(output_2, (2, 2), (2, 2), name="max_pooling_2")
            print(output_pool_2.name)

            output_3 = conv_layer(output_pool_2, (3, 3), 128, [1, 1, 1, 1], name="conv_3", batch_normalize=True,
                                  activation=lrelu)
            output_4 = conv_layer(output_3, (1, 1), 64, [1, 1, 1, 1], name="conv_4", batch_normalize=True,
                                  activation=lrelu)
            output_5 = conv_layer(output_4, (3, 3), 128, [1, 1, 1, 1], name="conv_5", batch_normalize=True,
                                  activation=lrelu)
            output_pool_3 = max_pooling(output_5, (2, 2), (2, 2), name="max_pooling_3")
            print(output_pool_3.name)

            output_6 = conv_layer(output_pool_3, (3, 3), 256, [1, 1, 1, 1], name="conv_6", batch_normalize=True,
                                  activation=lrelu)
            output_7 = conv_layer(output_6, (1, 1), 128, [1, 1, 1, 1], name="conv_7", batch_normalize=True,
                                  activation=lrelu)
            output_8 = conv_layer(output_7, (3, 3), 256, [1, 1, 1, 1], name="conv_8", batch_normalize=True,
                                  activation=lrelu)
            output_pool_4 = max_pooling(output_8, (2, 2), (2, 2), name="max_pooling_4")

            output_9 = conv_layer(output_pool_4, (3, 3), 512, [1, 1, 1, 1], name="conv_9", batch_normalize=True,
                                  activation=lrelu)
            output_10 = conv_layer(output_9, (1, 1), 256, [1, 1, 1, 1], name="conv_10", batch_normalize=True,
                                   activation=lrelu)
            output_11 = conv_layer(output_10, (3, 3), 512, [1, 1, 1, 1], name="conv_11", batch_normalize=True,
                                   activation=lrelu)
            output_12 = conv_layer(output_11, (1, 1), 256, [1, 1, 1, 1], name="conv_12", batch_normalize=True,
                                   activation=lrelu)
            output_13 = conv_layer(output_12, (3, 3), 512, [1, 1, 1, 1], name="conv_13", batch_normalize=True,
                                   activation=lrelu)
            output_pool_5 = max_pooling(output_13, (2, 2), (2, 2), name="max_pooling_5")

            shape = output_pool_5.get_shape()
            num_features = shape[1:4].num_elements()
            output_22 = tf.reshape(output_pool_5, [-1, num_features])
            # 55296=num_features
            fc_1 = fc_layer(output_22, num_features, 1024, name="fc_1", batch_normalize=True, activation=lrelu)
            fc_2 = fc_layer(fc_1, 1024, 2048, name="fc_2", batch_normalize=True, activation=lrelu)
            print(fc_2.name)
            fc_3 = fc_layer(fc_2, 2048, num_classes, name="fc_3", batch_normalize=False, activation=None)

            return fc_3

    def vggNet_2(self, input):
        with tf.name_scope("VGG16"):
            output_1 = conv_layer(input, (3, 3), 64, [1, 1, 1, 1], name="conv_1", batch_normalize=True, activation=lrelu)
            output_2 = conv_layer(output_1, (3, 3), 64, [1, 1, 1, 1], name="conv_2", batch_normalize=True,activation=lrelu)
            output_pool_1 = max_pooling(output_2, (2, 2), (2, 2), name="max_pooling_1")
            print(output_pool_1.get_shape())

            output_3 = conv_layer(output_pool_1, (3, 3), 128, [1, 1, 1, 1], name="conv_3", batch_normalize=True, activation=lrelu)
            output_4 = conv_layer(output_3, (3, 3), 128, [1, 1, 1, 1], name="conv_4", batch_normalize=True,activation=lrelu)
            output_pool_2 = max_pooling(output_4, (2, 2), (2, 2), name="max_pooling_2")
            print(output_pool_2.get_shape())

            output_5 = conv_layer(output_pool_2, (3, 3), 256, [1, 1, 1, 1], name="conv_5", batch_normalize=True,activation=lrelu)
            output_6 = conv_layer(output_5, (3, 3), 256, [1, 1, 1, 1], name="conv_6", batch_normalize=True,activation=lrelu)
            output_7 = conv_layer(output_6, (3, 3), 256, [1, 1, 1, 1], name="conv_7", batch_normalize=True,activation=lrelu)
            output_pool_3 = max_pooling(output_7, (2, 2), (2, 2), name="max_pooling_3")
            print(output_pool_2.get_shape())

            output_8 = conv_layer(output_pool_3, (3, 3), 512, [1, 1, 1, 1], name="conv_8", batch_normalize=True,activation=lrelu)
            output_9 = conv_layer(output_8, (3, 3), 512, [1, 1, 1, 1], name="conv_9", batch_normalize=True,activation=lrelu)
            output_10 = conv_layer(output_9, (3, 3), 512, [1, 1, 1, 1], name="conv_10", batch_normalize=True,activation=lrelu)
            output_pool_4 = max_pooling(output_10, (2, 2), (2, 2), name="max_pooling_4")
            print(output_pool_2.get_shape())

            output_11 = conv_layer(output_pool_4, (3, 3), 512, [1, 1, 1, 1], name="conv_11", batch_normalize=True,activation=lrelu)
            output_12 = conv_layer(output_11, (3, 3), 512, [1, 1, 1, 1], name="conv_12", batch_normalize=True,activation=lrelu)
            output_13 = conv_layer(output_12, (3, 3), 512, [1, 1, 1, 1], name="conv_13", batch_normalize=True,activation=lrelu)
            output_pool_5 = max_pooling(output_13, (2, 2), (2, 2), name="max_pooling_5")
            print(output_pool_2.get_shape())

            shape = output_pool_5.get_shape()
            num_features = shape[1:4].num_elements()
            output_22 = tf.reshape(output_pool_5, [-1, num_features])
            # 55296=num_features
            fc_1 = fc_layer(output_22, num_features, 1024, name="fc_1", batch_normalize=True, activation=lrelu)
            fc_2 = fc_layer(fc_1, 1024, num_classes, name="fc_2", batch_normalize=True, activation=lrelu)
            #print(fc_2.name)
            #fc_3 = fc_layer(fc_2, 2048, num_classes, name="fc_3", batch_normalize=False, activation=None)

            return fc_2


    def googLeNet(self, input):
        with tf.name_scope("GoogLeNet"):
            ouput_1 = conv_layer(input, kernel_size=(7, 7), out_channels=64, strides=[1, 2, 2, 1], name="conv_1",
                                 batch_normalize=False, activation=relu, training=True)
            output_pool_1 = max_pooling(ouput_1, (3, 3), (2, 2), name="max_pooling_1")

            output_2 = conv_layer(output_pool_1, (3, 3), 192, strides=[1, 1, 1, 1], name="conv_2",
                                  batch_normalize=False, activation=relu, training=True)
            output_pool_2 = max_pooling(output_2, (3, 3), (2, 2), name="max_pooling_2")

            inception_3a = inception_module(output_pool_2, 64, 96, 128, 16, 32, 32, name="inception_3a")
            inception_3b = inception_module(inception_3a, 128, 128, 192, 32, 96, 64, name="inception_3b")
            output_pool_3 = max_pooling(inception_3b, (3, 3), (2, 2), name="max_pooling_3")

            inception_4a = inception_module(output_pool_3, 192, 96, 208, 16, 48, 64, name="inception_4a")
            inception_4b = inception_module(inception_4a, 160, 112, 224, 24, 64, 64, name="inception_4b")
            inception_4c = inception_module(inception_4b, 128, 128, 256, 24, 64, 64, name="inception_4c")
            inception_4d = inception_module(inception_4c, 112, 144, 288, 32, 64, 64, name="inception_4d")
            inception_4e = inception_module(inception_4d, 256, 160, 320, 32, 128, 128, name="inception_4e")
            output_pool_4 = max_pooling(inception_4e, (3, 3), (2, 2), name="max_pooling_4")

            inception_5a = inception_module(output_pool_4, 256, 160, 320, 32, 128, 128, name="inception_5a")
            inception_5b = inception_module(inception_5a, 384, 192, 384, 48, 128, 128, name="inception_5b")
            shape = inception_5b.get_shape()
            print(shape)
            output_pool_5 = average_pooling(inception_5b, (shape[1], shape[2]), (1, 1), name="ave_pooling")

            dropout = tf.nn.dropout(x=output_pool_5, keep_prob=0.4)
            shape = dropout.get_shape()
            num_features = shape[1:4].num_elements()
            dropout = tf.reshape(dropout, [-1, num_features])
            linear = fc_layer(dropout, num_features, num_classes, name="fc_1", batch_normalize=False, activation=None, training=True)

            return linear


    def resNet(self, input):
        with tf.name_scope("ResNet"):
            output_1 = conv_layer(input, kernel_size=(7, 7), out_channels=64, strides=[1, 2, 2, 1],
                                  name="conv_1", batch_normalize=True, activation=relu, training=True)
            maxpooling = max_pooling(output_1, (2, 2), (2, 2), name="max_pooling")
            assert maxpooling.get_shape()[-1] == 64

            output_2 = building_block(maxpooling, (3, 3), 64, [1, 1, 1, 1], name="building_block1")
            output_3 = building_block(output_2, (3, 3), 64, [1, 1, 1, 1], name="building_block2")
            output_4 = building_block(output_3, (3, 3), 64, [1, 1, 1, 1], name="building_block3")

            output_5 = building_block(output_4, (3, 3), 128, [1, 2, 2, 1], name="building_block4")
            output_6 = building_block(output_5, (3, 3), 128, [1, 1, 1, 1], name="building_block5")
            output_7 = building_block(output_6, (3, 3), 128, [1, 1, 1, 1], name="building_block6")
            output_8 = building_block(output_7, (3, 3), 128, [1, 1, 1, 1], name="building_block7")

            output_9 = building_block(output_8, (3, 3), 256, [1, 2, 2, 1], name="building_block8")
            output_10 = building_block(output_9, (3, 3), 256, [1, 1, 1, 1], name="building_block9")
            output_11 = building_block(output_10, (3, 3), 256, [1, 1, 1, 1], name="building_block10")
            output_12 = building_block(output_11, (3, 3), 256, [1, 1, 1, 1], name="building_block11")
            output_13 = building_block(output_12, (3, 3), 256, [1, 1, 1, 1], name="building_block12")
            output_14 = building_block(output_13, (3, 3), 256, [1, 1, 1, 1], name="building_block13")

            output_15 = building_block(output_14, (3, 3), 512, [1, 2, 2, 1], name="building_block14")
            output_16 = building_block(output_15, (3, 3), 512, [1, 1, 1, 1], name="building_block15")
            output_17 = building_block(output_16, (3, 3), 512, [1, 1, 1, 1], name="building_block16")

            shape_output_17 = output_17.get_shape()

            avepooling = average_pooling(output_17, (shape_output_17[1], shape_output_17[2]), (1, 1), name="ave_pooling")

            shape_avepooling = avepooling.get_shape()
            num_features = shape_avepooling[1:4].num_elements()
            avepooling = tf.reshape(avepooling, [-1, num_features])
            linear = fc_layer(avepooling, num_features, num_classes, name="fc_1", batch_normalize=False,
                              activation=None, training=True)

            return linear

    def train(self):
        # read data from the file of train.tfrecords
        filename = r"D:\CNN\training_data.tfrecords"
        id, ld = read_example(filename, batch_size)
        id = tf.image.rot90(id, k=2)
        IMAGE_HEIGHT = 288
        IMAGE_WIDTH = 384
        IMAGE_DEPTH = 1
        # set the placeholder of input image and label
        image = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH],
                               name="image_placeholder")
        label = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name="label_placeholder")

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 2, 0.98, staircase=True)

        #model_output = SelectModel(input=image)
        with tf.variable_scope('net'):
            output = self.vggNet_2(input=image)
            print(output.name)
        train_accuracy = calculate_accuracy(output, label)
        with tf.name_scope("loss"):
            # use softmax_cross_entropy_with_logits(), so labels must be one_hot coding.
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,
                                                                             labels=label,
                                                                             name="cross_entropy"), name="loss")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8).minimize(loss, global_step=global_step)
        # initial the parameters
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        # Start the operation of saving the fine model
        saver = tf.train.Saver()
        train_loss_list = []
        f1 = open(r"D:\CNN\vgg16_loss", "w")
        f2 = open(r"D:\CNN\vgg16_error", "w")
        start_time = time.time()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        data_path = r"D:\CNN\data_training"
        data_num = count_numbers(data_path)
        batch_nums = int(math.ceil(data_num / batch_size))
        for epoch in range(epoch_num):
            for batch_num in range(batch_nums):
                image_data, label_data = sess.run([id, ld])
                _, output_data, loss_data, accuracy = sess.run([train_step, output, loss, train_accuracy],
                                                               feed_dict={image: image_data, label: label_data})
            train_loss_list.append([epoch, loss_data])
            print("epoch:{}...".format(epoch),
                  "average_time: {:.2f}...".format((time.time() - start_time) / (epoch + 1)),
                  "total_time: {:.2f}...".format(time.time() - start_time),
                  "loss:{}...".format(loss_data),
                  "error:{:.3f}%...".format(100 - accuracy))
            if (epoch + 1) % save_rate == 0 or (epoch + 1) == epoch_num:
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                saver.save(sess, os.path.join(model_path, 'fine_parameters'), global_step=epoch + 1)
            f1.write(str(epoch + 1) + "\t " + str(loss_data) + "\n")
            f2.write(str(epoch + 1) + "\t" + str(100 - accuracy) + "\n")
        draw_loss(train_loss_list)
        coord.request_stop()
        coord.join(threads)
        f1.close()
        f2.close()
        sess.close()

if __name__ == "__main__":
    num_classes = 2
    learning_rate_base = 0.001
    save_rate = 4
    epoch_num = 20
    batch_size = 64
    model_path = "./VGG16"
    model_output = SelectModel(num_classes=2,
                               learning_rate_base=0.001,
                               save_rate=4,
                               epoch_num=40,
                               batch_size=64,
                               model_path="./VGG16")
    model_output.train()
