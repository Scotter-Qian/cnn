# make tf.record 文件
# 运行时只需要修改下路径
import os
import pickle
import tensorflow as tf
import numpy as np
import math

#定义函数搜索要训练的数据路径
def search_file(path):
    file = list()
    for name in os.listdir(path):
        name_path = os.path.join(path, name)
        if os.path.isfile(name_path):
            if name_path.endswith(".xy"):
                file.append(name_path)
    return np.array(file)


def encode_to_tfrecords(path):
    #创建对象，用于记录文件写入记录；train时将"valid.tfrecord"改为"train.tfrecord"
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    data_path_file = search_file(path)
    for file_path in data_path_file:
        file = open(file_path, "rb")
        data = pickle.load(file)
        image = (data["x"])
        image = np.reshape(image, (288, 384, 1))
        label = (data["y"])
        label = np.reshape(label, 1)
        #将图片转换成原生bytes
        image_raw = image.tobytes()
        #label_raw = label.tobytes()
        #将数据整理成TFRecord需要的数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
        #序列化写入文件
        writer.write(example.SerializeToString())
    writer.close()
    print("tfrecord_file is done!")

#读取TFRecord格式文件，返回读取到的batch_size张图片以及对应的标签
#filename:TFRecord格式文件路径
def read_example(filename, batch_size):
    #创建文件读取器，从队列文件中读取数据
    reader = tf.TFRecordReader()
    #创建文件名队列
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    _, serialized_example = reader.read(filename_queue)
    #每次读取batch_size张图片,如果使用tf.train.shuffle_batch,则不一样
    min_after_dequeue = 10
    batch = tf.train.shuffle_batch([serialized_example],
                            batch_size=batch_size,
                            capacity=min_after_dequeue+2000+3 * batch_size,
                            min_after_dequeue=10,
                            num_threads=2)
    parsed_example = tf.parse_example(batch, features={'image': tf.FixedLenFeature([], tf.string),
                                                       'label': tf.FixedLenFeature([], tf.int64)})
    image_raw = tf.decode_raw(parsed_example['image'], tf.uint8)
    #IMAGE_HEIGHT为288，IMAGE_WIDTH为384, IMAGE_DEPTH为1
    image = tf.cast(tf.reshape(image_raw, [batch_size, 288, 384, 1]), tf.float32)
    image = image/255.0
    label_raw = tf.cast(parsed_example['label'], tf.int32)
    label = tf.reshape(label_raw, [batch_size*1])
    #depth=num_classes,这里是二分类，所以num_classes=2
    label = tf.one_hot(label, depth=2)
    return image, label

def get_example_nums(tfrecord_filenames):
    nums = 0
    for record in tf.python_io.tf_record_iterator(tfrecord_filenames):
        nums+=1
    return nums


if __name__ == "__main__":
    epoch = 3
    path = r"E:\CNN\data\train"
    encode_to_tfrecords(path)

    """
    filename = r"D:\gan\train.tfrecords"
    batch_size = 3
    ima, lab = read_example(filename, batch_size)

    numbers = get_example_nums(filename)
    print(numbers)
    batch_nums = int(math.ceil(numbers/batch_size))
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    #创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    #启动填充队列的线程，此时文件名才开始进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for j in range(epoch):
        for i in range(batch_nums):
            image, label = sess.run([ima, lab])
            print(label)
    coord.request_stop()
    coord.join(threads)
    sess.close()
    """