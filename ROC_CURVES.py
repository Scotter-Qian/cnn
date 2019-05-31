import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import math
import os


def read_example(filename, batch_size):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    _, serialized_example = reader.read(filename_queue)
    min_queue_examples = 500
    batch = tf.train.shuffle_batch([serialized_example], batch_size=batch_size,
                                   capacity=min_queue_examples + 100 * batch_size, min_after_dequeue=min_queue_examples,
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

def TP_FN_FP(label, pre):
    TP = np.sum(np.logical_and(pre == 1, label == 1))
    FP = np.sum(np.logical_and(pre == 1, label == 0))
    FN = np.sum(np.logical_and(pre == 0, label == 1))
    TN = np.sum(np.logical_and(pre == 0, label == 0))
    TPR = TP/(TP+FN)
    FPR = FP/(TN+FP)
    return TPR, FPR

def draw_roc_curve(FPR, TPR):
    #FPR, TPR, thresholds = roc_curve(true, pre, pos_label=2)
    AUC = auc(FPR, TPR)
    plt.figure(figsize=(8,6))
    plt.plot(FPR[:], TPR[:], "b", linewidth=2.0, label="AUC = %0.5f"%AUC)
    plt.plot((0, 1), (0, 1), linestyle="--", linewidth=2.0)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title("ROC Curve", fontsize=18)
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    #plt.grid(True)
    plt.legend()  # 显示label
    plt.show()

def count_numbers(data_path):
    num = 0
    for name in os.listdir(data_path):
        if name.endswith(".xy"):
            num += 1

    return num


def test():
    #文件读取
    BATCH_SIZE = 64
    filename = r"D:\CNN\test.tfrecords"
    ima, lab = read_example(filename, BATCH_SIZE)
    #加载模型
    sess = tf.Session()
    saver = tf.train.import_meta_graph(r"D:\CNN\GoogLeNet\fine_parameters-27.meta")
    saver.restore(sess, r"D:\CNN\GoogLeNet\fine_parameters-27")
    graph = tf.get_default_graph()
    #retrieve tensors, operations, ect
    image = graph.get_tensor_by_name("image_placeholder:0")
    label = graph.get_tensor_by_name("label_placeholder:0")
    output = graph.get_tensor_by_name("net/GoogLeNet/fc_1/add:0")
    #net/ResNet/fc_1/add:0
    #net/ResNet/fc_1/add:0
    #net/GoogLeNet/fc_1/add:0
    #net/VGG16/fc_3/add:0
    #计算测试样本数
    test_path = r"D:\CNN\data\test"
    num_test = count_numbers(test_path)

    #计算迭代次数，向上取整
    num_step = int(math.ceil(num_test/BATCH_SIZE))
    print("num_step: ", num_step)
    #重新计算实际测试样本数
    num_example = num_step * BATCH_SIZE
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    output_set = []
    label_set = []
    for i in range(num_step):
        image_data, label_data = sess.run([ima, lab])
        #print(image_data, label_data)
        pre = sess.run(tf.nn.softmax(output), feed_dict={image:image_data, label:label_data})
        print(pre)
        index_label = np.argmax(label_data, axis=1)
        label_set.append(index_label)
        index_pre = np.argmax(pre, axis=1)
        for j in range(BATCH_SIZE):
            output_set.append(pre[j][index_pre[j]])
    output_set = np.reshape(np.array(output_set), num_example)
    _output_set = sorted(output_set, reverse=True)
    label_set = np.reshape(np.array(label_set), num_example)
    f = open(r"D:\CNN\Roc_Curve(GoogLeNet)", "w")
    FPR_SET = []
    TPR_SET = []
    for k in _output_set:
        y_pre = (output_set>k).astype(int)
        TPR, FPR = TP_FN_FP(label_set, y_pre)
        f.write(str(TPR) + "\t" + str(FPR) + "\n")
        FPR_SET.append(FPR)
        TPR_SET.append(TPR)
    FPR_SET = np.array(FPR_SET)
    TPR_SET = np.array(TPR_SET)
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    draw_roc_curve(FPR_SET, TPR_SET)
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__=="__main__":
    test()
