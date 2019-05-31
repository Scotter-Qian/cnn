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
        writer = tf.python_io.TFRecordWriter("test.tfrecords")
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
#定义构建网络模型操作函数
    def lrelu(self, input):
        return tf.nn.leaky_relu(input, name='leaky_relu')

    def max_pooling(self, input, size, stride, name):
        with tf.variable_scope(name):
            x = tf.layers.max_pooling2d(input, size, stride, padding="SAME", name=name)
        return x

    def batch_normalization(self, input, training):
        shape = input.get_shape().as_list()
        dimension = shape[-1]
        if len(shape) == 4:
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2])
        else:
            mean, variance = tf.nn.moments(input, axes=[0])
        beta = tf.get_variable("beta", dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32),
                               trainable=training)
        gamma = tf.get_variable("gamma", dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32),
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
                y = self.batch_normalization(x, training)
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
                fc = self.batch_normalization(fc, training)
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
#定义模型框架函数
    def vggNet(self, input):
        with tf.name_scope("VGG11"):
            output_1 = self.conv_layer(input, (3, 3), 32, [1, 1, 1, 1], name="conv_1", batch_normalize=True,
                                  activation=self.lrelu)#第一次卷积输出的特征图为32，如果设为64就出现OOM错误
            output_pool_1 = self.max_pooling(output_1, (2, 2), (2, 2), name="max_pooling_1")
            print(output_pool_1.get_shape())

            output_2 = self.conv_layer(output_pool_1, (3, 3), 64, [1, 1, 1, 1], name="conv_2", batch_normalize=True,
                                  activation=self.lrelu)
            output_pool_2 = self.max_pooling(output_2, (2, 2), (2, 2), name="max_pooling_2")
            print(output_pool_2.get_shape())

            output_3 = self.conv_layer(output_pool_2, (3, 3), 128, [1, 1, 1, 1], name="conv_3", batch_normalize=True,
                                  activation=self.lrelu)
            output_4 = self.conv_layer(output_3, (3, 3), 128, [1, 1, 1, 1], name="conv_5", batch_normalize=True,
                                  activation=self.lrelu)
            output_pool_3 = self.max_pooling(output_4, (2, 2), (2, 2), name="max_pooling_3")
            print(output_pool_3.get_shape())

            output_5 = self.conv_layer(output_pool_3, (3, 3), 256, [1, 1, 1, 1], name="conv_7", batch_normalize=True,
                                  activation=self.lrelu)
            output_6 = self.conv_layer(output_5, (3, 3), 256, [1, 1, 1, 1], name="conv_8", batch_normalize=True,
                                  activation=self.lrelu)
            output_pool_4 = self.max_pooling(output_6, (2, 2), (2, 2), name="max_pooling_4")
            print(output_pool_4.get_shape())

            output_7 = self.conv_layer(output_pool_4, (3, 3), 256, [1, 1, 1, 1], name="conv_12", batch_normalize=True,
                                   activation=self.lrelu)
            output_8 = self.conv_layer(output_7, (3, 3), 256, [1, 1, 1, 1], name="conv_13", batch_normalize=True,
                                   activation=self.lrelu)
            output_pool_5 = self.max_pooling(output_8, (2, 2), (2, 2), name="max_pooling_5")
            print(output_pool_5.get_shape())

            shape = output_pool_5.get_shape()
            num_features = shape[1:4].num_elements()
            output_ = tf.reshape(output_pool_5, [-1, num_features])
            # 55296=num_features
            fc_1 = self.fc_layer(output_, num_features, 512, name="fc_1", batch_normalize=True, activation=self.lrelu)
            fc_2 = self.fc_layer(fc_1, 512, 1024, name="fc_2", batch_normalize=True, activation=self.lrelu)
            print(fc_2.name)
            fc_3 = self.fc_layer(fc_2, 1024, self.num_classes, name="fc_3", batch_normalize=False, activation=None)
            return fc_3

#-----------------------------------------------------------------------------------------------------------------------
#定义训练函数
    def train(self):
        # 从train.tfrecords中读取数据
        train_data = r"E:\CNN\train.tfrecords"
        test_data = r"E:\CNN\test.tfrecords"
        train_ima, train_lab = self.read_example(train_data, self.batch_size)
        test_ima, test_lab = self.read_example(test_data, self.batch_size)


        # 占位符
        IMAGE_HEIGHT = 288
        IMAGE_WIDTH = 384
        IMAGE_DEPTH = 1
        image = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH],
                               name="image_placeholder")
        label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name="label_placeholder")

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate_base, global_step, 2, 0.98, staircase=True)

        # model_output = SelectModel(input=image)
        with tf.variable_scope('net'):
            output = self.vggNet(input=image)
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
        f1 = open(r"E:\CNN\VGGNet11_train_loss", "w")
        f2 = open(r"E:\CNN\VGGNet11_train_error", "w")
        f3 = open(r"E:\CNN\VGGNet11_test_error", "w")
        start_time = time.time()
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动QueueRunner,此时文件名队列已经进队

        data_path = r"E:\CNN\data\train"
        data_num = self.count_numbers(data_path)
        batch_nums = int(math.ceil(data_num / self.batch_size))
        print(data_num, batch_nums)
        for epoch in range(self.epoch_num):
            avg_loss = 0
            train_avg_accuracy = 0
            test_avg_accuracy = 0
            for batch_num in range(batch_nums):
                train_imas, train_labs = sess.run([train_ima, train_lab])
                test_imas, test_labs = sess.run([test_ima, test_lab])
                _, loss_data, train_accuracy = sess.run([train_step, loss, accuracy],
                                                        feed_dict={image: train_imas, label: train_labs})
                test_accuracy = sess.run(accuracy, feed_dict={image: test_imas, label: test_labs})
                avg_loss += loss_data / batch_nums
                train_avg_accuracy += train_accuracy / batch_nums
                test_avg_accuracy += test_accuracy / batch_nums
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
            if (epoch + 1) % self.save_rate == 0:
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
                               epoch_num=32,
                               batch_size=40,
                               model_path="./VGG11")#batch_size 设置为64时会出现OOM错误

    import time, datetime

    startTime = datetime.datetime(2019, 3, 4, 16, 40)
    print("Program not starting yet...")
    while datetime.datetime.now() < startTime:
        time.sleep(1)
    print("program now strarts on %s"%startTime)
    model_output.train()

