import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
import pickle


class SelectModel(object):
    def __init__(self, num_classes, learning_rate_base, save_rate, epoch_num, batch_size, model_path):
        self.num_classes = num_classes
        self.learning_rate_base = learning_rate_base
        self.save_rate = save_rate
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.model_path = model_path

#-----------------------------------------------------------------------------------------------------------------------
#定义从tf.record文件中读取数据函数
    # 定义函数搜索要训练的数据路径
    def search_file(self, path):
        file = list()
        for name in os.listdir(path):
            name_path = os.path.join(path, name)
            if os.path.isfile(name_path):
                if name_path.endswith(".xy"):
                    file.append(name_path)
        return np.array(file)

    def encode_to_tfrecords(self, path):
        # 创建对象，用于记录文件写入记录；train时将"valid.tfrecord"改为"train.tfrecord"
        writer = tf.python_io.TFRecordWriter("train.tfrecords")
        data_path_file = self.search_file(path)
        for file_path in data_path_file:
            file = open(file_path, "rb")
            data = pickle.load(file)
            image = (data["x"])
            image = np.reshape(image, (288, 384, 1))
            label = (data["y"])
            label = np.reshape(label, 1)
            # 将图片转换成原生bytes
            image_raw = image.tobytes()
            # label_raw = label.tobytes()
            # 将数据整理成TFRecord需要的数据结构
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
            # 序列化写入文件
            writer.write(example.SerializeToString())
        writer.close()
        print("tfrecord_file is done!")

    # 读取TFRecord格式文件，返回读取到的batch_size张图片以及对应的标签
    # filename:TFRecord格式文件路径
    def read_example(self, filename, batch_size):
        # 创建文件读取器，从队列文件中读取数据
        reader = tf.TFRecordReader()
        # 创建文件名队列
        filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
        _, serialized_example = reader.read(filename_queue)
        # 每次读取batch_size张图片,如果使用tf.train.batch,则不一样
        min_after_dequeue = 100
        batch = tf.train.shuffle_batch([serialized_example],
                                       batch_size=batch_size,
                                       capacity=min_after_dequeue + 2000 + 3 * batch_size,
                                       min_after_dequeue=min_after_dequeue,
                                       num_threads=2)
        parsed_example = tf.parse_example(batch, features={'image': tf.FixedLenFeature([], tf.string),
                                                           'label': tf.FixedLenFeature([], tf.int64)})
        image_raw = tf.decode_raw(parsed_example['image'], tf.uint8)
        # IMAGE_HEIGHT为288，IMAGE_WIDTH为384, IMAGE_DEPTH为1
        image = tf.cast(tf.reshape(image_raw, [batch_size, 288, 384, 1]), tf.float32)
        image = image / 255.0
        label_raw = tf.cast(parsed_example['label'], tf.int32)
        label = tf.reshape(label_raw, [batch_size * 1])
        # depth=num_classes,这里是二分类，所以num_classes=2
        label = tf.one_hot(label, depth=2)
        return image, label
#-----------------------------------------------------------------------------------------------------------------------
#定义网络操作函数
    def lrelu(self, input):
        return tf.nn.leaky_relu(input, name='leaky_relu')

    def relu(self, input):
        return tf.nn.relu(input, name="relu")

    def max_pooling(self, input, size, stride, name):
        with tf.variable_scope(name):
            x = tf.layers.max_pooling2d(input, size, stride, padding="SAME", name=name)
        return x

    def average_pooling(self, input, size, strides, name):
        with tf.variable_scope(name):
            x = tf.layers.average_pooling2d(input, size, strides, padding="SAME", name=name)
        return x

    def batch_normalization(self, input, training, name):
        with tf.name_scope(name):
            shape = input.get_shape().as_list()
            dimension = shape[-1]
            if len(shape) == 4:
                mean, variance = tf.nn.moments(input, axes=[0, 1, 2])
            else:
                mean, variance = tf.nn.moments(input, axes=[0])
            beta = tf.get_variable("beta", dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32),
                                   trainable=training)
            gamma = tf.get_variable("gamma", dimension, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32),
                                    trainable=training)
            bn = tf.nn.batch_normalization(input, mean, variance, beta, gamma, variance_epsilon=0.001)
            return bn

    def conv_layer(self, input, kernel_size, out_channels, strides, name, batch_normalize, activation, training=True):
        in_channels = input.get_shape()[-1]
        with tf.variable_scope(name):
            w = tf.get_variable(name="weights", shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=training)
            b = tf.get_variable(name="biases", shape=[out_channels],
                                dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=training)
            x = tf.nn.conv2d(input, w, strides, padding="SAME", name="conv")
            x = tf.nn.bias_add(x, b, name="bias_add")
            # 如果有BN层，则BN层应放在tf.layers.conv2d和activation之间
            if batch_normalize:
                y = self.batch_normalization(x, training, name="conv_bn")
                if activation:
                    return activation(y)
                else:
                    return y
            else:
                if activation:
                    return activation(x)
                else:
                    return x

    def fc_layer(self, input, in_size, out_size, name, batch_normalize, activation, training=True):
        with tf.variable_scope(name):
            weights = tf.get_variable('fc_weights', shape=[in_size, out_size],
                                      initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
            biases = tf.get_variable('fc_bias', shape=[out_size],
                                     initializer=tf.constant_initializer(0.0), trainable=training)
            fc = tf.matmul(input, weights) + biases
            if batch_normalize:
                fc = self.batch_normalization(fc, training, name="fc_bn")
            if activation:
                fc = activation(fc)

            return fc

    def calculate_accuracy(self, logits, labels):
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0

        return accuracy

    def draw(self, data, title, name):
        data = np.array(data)
        plt.figure(figsize=[8, 6])
        plt.plot(data[:, 0], data[:, 1], 'b', linewidth=2.0)
        plt.title(title, fontsize=18)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel(name, fontsize=16)
        plt.show()

    def count_numbers(self, data_path):
        num = 0
        for name in os.listdir(data_path):
            if name.endswith(".xy"):
                num += 1

        return num

#-----------------------------------------------------------------------------------------------------------------------
#定义ResNet的inception模块
    def building_block(self, inputs, filter_size, filters, strides, name):
        with tf.variable_scope(name):
            x = self.cbam_block(inputs, name="resNet_cbam")
            if strides[1] > 1:
                x = self.conv_layer(x, (1, 1), filters, strides, name="conv_x", batch_normalize=True, activation=False,
                                    training=True)
            f_x = self.conv_layer(inputs, filter_size, filters, strides=strides, name="conv_1", batch_normalize=True,
                                  activation=self.relu, training=True)
            f_x = self.conv_layer(f_x, filter_size, filters, strides=[1, 1, 1, 1], name="conv_2", batch_normalize=True,
                                  activation=False, training=True)
            f_x = tf.add(f_x, x)
            output = tf.nn.relu(f_x)

            return output

#-----------------------------------------------------------------------------------------------------------------------
#定义CBAM的channel attention模块和spatial attention模块
    def cbam_block(self, input_feature, name, ratio=8):
        with tf.variable_scope(name):
            attention_feature = self.channel_attention(input_feature, "ch_at", ratio)
            attention_feature = self.spatial_attention(attention_feature, "sp_at")
            print("CBAM Hello")
        return attention_feature

    def channel_attention(self, input_feature, name, ratio=8):
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)
        with tf.variable_scope(name):
            channel = input_feature.get_shape()[-1]
            print(channel)
            avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)
            assert avg_pool.get_shape()[1:] == (1, 1, channel)
            avg_pool = tf.layers.dense(inputs=avg_pool,
                                       units=channel // ratio,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       name="mlp_0",
                                       reuse=None)
            assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
            avg_pool = tf.layers.dense(inputs=avg_pool,
                                       units=channel,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       name="mlp_1",
                                       reuse=None)
            assert avg_pool.get_shape()[1:] == (1, 1, channel)

            max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel)
            max_pool = tf.layers.dense(inputs=max_pool,
                                       units=channel // ratio,
                                       activation=tf.nn.relu,
                                       name="mlp_0",
                                       reuse=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
            max_pool = tf.layers.dense(inputs=max_pool,
                                       units=channel,
                                       name="mlp_1",
                                       reuse=True)
            assert max_pool.get_shape()[1:] == (1, 1, channel)

            scale = tf.sigmoid(avg_pool + max_pool, "sigmoid")
            return input_feature * scale

    def spatial_attention(self, input_feature, name):
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
                                      strides=[1, 1],
                                      padding="same",
                                      activation=None,
                                      kernel_initializer=kernel_initializer,
                                      use_bias=False,
                                      name="conv")
            assert concat.get_shape()[-1] == 1
            concat = tf.sigmoid(concat, "sigmoid")

            return input_feature * concat

#-----------------------------------------------------------------------------------------------------------------------
#定义模型函数
    def resNet(self, input):
        with tf.name_scope("AM-ResNet34"):
            output_1 = self.conv_layer(input, kernel_size=(7, 7), out_channels=64, strides=[1, 2, 2, 1],
                                       name="conv_1", batch_normalize=True, activation=self.relu, training=True)
            maxpooling = self.max_pooling(output_1, (2, 2), (2, 2), name="max_pooling")
            assert maxpooling.get_shape()[-1] == 64

            output_2 = self.building_block(maxpooling, (3, 3), 64, [1, 1, 1, 1], name="building_block1")
            output_3 = self.building_block(output_2, (3, 3), 64, [1, 1, 1, 1], name="building_block2")
            output_4 = self.building_block(output_3, (3, 3), 64, [1, 1, 1, 1], name="building_block3")

            output_5 = self.building_block(output_4, (3, 3), 128, [1, 2, 2, 1], name="building_block4")
            output_6 = self.building_block(output_5, (3, 3), 128, [1, 1, 1, 1], name="building_block5")
            output_7 = self.building_block(output_6, (3, 3), 128, [1, 1, 1, 1], name="building_block6")
            output_8 = self.building_block(output_7, (3, 3), 128, [1, 1, 1, 1], name="building_block7")

            output_9 = self.building_block(output_8, (3, 3), 256, [1, 2, 2, 1], name="building_block8")
            output_10 = self.building_block(output_9, (3, 3), 256, [1, 1, 1, 1], name="building_block9")
            output_11 = self.building_block(output_10, (3, 3), 256, [1, 1, 1, 1], name="building_block10")
            output_12 = self.building_block(output_11, (3, 3), 256, [1, 1, 1, 1], name="building_block11")
            output_13 = self.building_block(output_12, (3, 3), 256, [1, 1, 1, 1], name="building_block12")
            output_14 = self.building_block(output_13, (3, 3), 256, [1, 1, 1, 1], name="building_block13")

            output_15 = self.building_block(output_14, (3, 3), 512, [1, 2, 2, 1], name="building_block14")
            output_16 = self.building_block(output_15, (3, 3), 512, [1, 1, 1, 1], name="building_block15")
            output_17 = self.building_block(output_16, (3, 3), 512, [1, 1, 1, 1], name="building_block16")

            shape_output_17 = output_17.get_shape()
            avepooling = self.average_pooling(output_17, (shape_output_17[1], shape_output_17[2]), (1, 1),
                                              name="ave_pooling")

            shape_avepooling = avepooling.get_shape()
            num_features = shape_avepooling[1:4].num_elements()
            avepooling = tf.reshape(avepooling, [-1, num_features])
            linear = self.fc_layer(avepooling, num_features, self.num_classes, name="fc_1", batch_normalize=False,
                              activation=None, training=True)


            return linear

#-----------------------------------------------------------------------------------------------------------------------
#定义训练函数
    def train(self):
        # 从train.tfrecords中读取数据
        train_data = r"D:\CNN\train.tfrecords"
        test_data = r"D:\CNN\test.tfrecords"
        train_ima, train_lab = self.read_example(train_data, self.batch_size)
        test_ima, test_lab = self.read_example(test_data, self.batch_size)
        IMAGE_HEIGHT = 288
        IMAGE_WIDTH = 384
        IMAGE_DEPTH = 1
        # 占位符
        image = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH],
                               name="image_placeholder")
        label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name="label_placeholder")

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate_base, global_step, 2, 0.98, staircase=True)

        # model_output = SelectModel(input=image)
        with tf.variable_scope('net'):
            output = self.resNet(input=image)
            print(output.name)
        accuracy = self.calculate_accuracy(output, label)
        with tf.name_scope("loss"):
            # use softmax_cross_entropy_with_logits(), so labels must be one_hot coding.
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,
                                                                             labels=label,
                                                                             name="cross_entropy"), name="loss")
        # 训练时需要添加一下几行，这样才能计算平均值和标准差的滑动平均，
        # 输入参数training=True.测试时，输入参数training=False,他就没了。
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8).minimize(loss, global_step=global_step)
        # 初始化
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        # 启动保存模型操作
        saver = tf.train.Saver()
        train_loss_list = []
        train_accuracy_list = []
        f1 = open(r"E:\CNN\resNet34CBAM_train_loss", "w")
        f2 = open(r"E:\CNN\resNet34CBAM_train_error", "w")
        f3 = open(r"E:\CNN\resNet34CBAM_test_error", "w")
        start_time = time.time()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        data_path = r"D:\CNN\data\train"
        data_num = self.count_numbers(data_path)
        batch_nums = int(math.ceil(data_num / self.batch_size))
        print(data_num,batch_nums)
        for epoch in range(self.epoch_num):
            avg_loss = 0
            train_avg_accuracy = 0
            test_avg_accuracy = 0
            for batch_num in range(batch_nums):
                train_imas, train_labs = sess.run([train_ima, train_lab])
                test_imas, test_labs = sess.run([test_ima, test_lab])
                _, loss_data, train_accuracy = sess.run([train_step, loss, accuracy],
                                                        feed_dict={image: train_imas, label:train_labs})
                test_accuracy = sess.run(accuracy, feed_dict={image: test_imas, label: test_labs})
                avg_loss += loss_data/batch_nums
                train_avg_accuracy += train_accuracy/batch_nums
                test_avg_accuracy += test_accuracy/batch_nums
            train_loss_list.append([epoch, avg_loss])
            train_accuracy_list.append([epoch, 100 - train_avg_accuracy])
            f1.write(str(epoch + 1) + "\t " + str(avg_loss) + "\n")
            f2.write(str(epoch + 1) + "\t" + str(100 - train_avg_accuracy) + "\n")
            f3.write(str(epoch + 1) + "\t" + str(100 - test_avg_accuracy) + "\n")
            print("epoch:{}...".format(epoch),
                  "average_time: {:.2f}...".format((time.time() - start_time) / (epoch + 1)),
                  "total_time: {:.2f}...".format(time.time() - start_time),
                  "loss:{}...".format(avg_loss),
                  "train_error:{:.3f}%...".format(100 - train_avg_accuracy),
                  "test_error:{:.3f}%...".format(100 - test_avg_accuracy))
            if (epoch + 1) % self.save_rate == 0 or (epoch + 1) == self.epoch_num:
                if not os.path.exists(self.model_path):
                    os.makedirs(self.model_path)
                saver.save(sess, os.path.join(self.model_path, 'fine_parameters'), global_step=epoch + 1)

        self.draw(train_loss_list, "Train Loss Curve", "Train Loss")
        self.draw(train_accuracy_list, "Train Error Curve", "Train Error")
        coord.request_stop()
        coord.join(threads)
        f1.close()
        f2.close()
        f3.close()
        sess.close()


if __name__ == "__main__":
    model_output = SelectModel(num_classes=2,
                               learning_rate_base=0.001,
                               save_rate=1,
                               epoch_num=30,
                               batch_size=64,
                               model_path="./ResNet34CBAM")
    import time, datetime

    startTime = datetime.datetime(2019, 3, 27, 16, 52)
    print("Program not starting yet...")
    while datetime.datetime.now() < startTime:
        time.sleep(1)
    print("program now strarts on %s"%startTime)
    model_output.train()