import tensorflow as tf
import numpy as np
import os
import pickle
import cv2
from data_read import *
import time

def count_number(filename):
    num = 0
    for name in tf.python_io.tf_record_iterator(filename):
            num += 1

    return num


def test():
    BATCH_SIZE = 64
    filename = r"E:\CNN\test.tfrecords"
    ima, lab = read_example(filename, BATCH_SIZE)

    num_test = count_number(filename)
    num_step = int(math.ceil(num_test / BATCH_SIZE))
    num_example = num_step * BATCH_SIZE
    print("num_test:", num_test, "num_examples:", num_example)

    # restore the model
    sess = tf.Session()
    saver = tf.train.import_meta_graph("E:\\CNN\\AM-VGG16\\fine_parameters-30.meta")
    #ckpt = tf.train.latest_checkpoint("E:\\CNN\\AM-VGG16")
    saver.restore(sess, "E:\\CNN\\AM-VGG16\\fine_parameters-30")
    #saver.restore(sess, ckpt)
    graph = tf.get_default_graph()
    # retrieve tensors, operations, ect
    image = graph.get_tensor_by_name("image_placeholder:0")
    label = graph.get_tensor_by_name("label_placeholder:0")
    output = graph.get_tensor_by_name("net/AM-VGG16/fc_3/add:0")
    
    #net/AM-ResNet34/fc_1/add:0
    #net/AM-VGG16/fc_3/add:0

    if not os.path.exists("./TP"):
        os.mkdir("./TP")
    if not os.path.exists("./TN"):
        os.mkdir("./TN")
    if not os.path.exists("./FP"):
        os.mkdir("./FP")
    if not os.path.exists("./FN"):
        os.mkdir("./FN")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    start_time = time.time()
    for i in range(num_step):
        image_data, label_data = sess.run([ima, lab])
        pre = sess.run(tf.nn.softmax(output), feed_dict={image: image_data, label:label_data})
        print(pre)
        index_pre = np.argmax(pre, axis=1)
        print(index_pre)
        label_data = np.argmax(label_data, axis=1)
        print(label_data)
        """
        for j in range(BATCH_SIZE):
            if np.logical_and(index_pre[j] == 1, label_data[j] == 1):
                cv2.imwrite("./TP/%d_%d.png" %(i,j), np.reshape(image_data[j]*255.0, (288, 384)))
            if np.logical_and(index_pre[j] == 0, label_data[j] == 0):
                cv2.imwrite("./TN/%d_%d.png"%(i,j), np.reshape(image_data[j]*255.0, (288, 384)))
            if np.logical_and(index_pre[j] == 1, label_data[j] == 0):
                cv2.imwrite("./FP/%d_%d.png" %(i,j), np.reshape(image_data[j]*255.0, (288, 384)))
            if np.logical_and(index_pre[j] == 0, label_data[j] == 1):
                cv2.imwrite("./FN/%d_%d.png" %(i,j), np.reshape(image_data[j]*255.0, (288, 384)))
        """
    end_time = time.time()
    print((end_time - start_time)/(num_step*BATCH_SIZE))

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == "__main__":
    test()



















