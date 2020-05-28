from data_read import *
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt
import math

def count_numbers(filename):
    num = 0
    for name in tf.python_io.tf_record_iterator(filename):
            num += 1

    return num

def test():
    # read data from tf.record
    BATCH_SIZE = 64
    # path = r"D:\vgg\data\valid"
    # encode_to_tfrecords(path)
    filename = r"E:\CNN\combined_testing_data.tfrecords"
    ima, lab = read_example(filename, BATCH_SIZE)
    """
    # Calculate the number of test samples
    test_path = r"D:\vgg\new_data\valid"
    file_path_set = search_file(test_path)
    num_test = len(file_path_set)
    # Calculate the number of iterations, round up
    """

    num_test = count_numbers(filename)
    num_step = int(math.ceil(num_test / BATCH_SIZE))
    # Recalculate the actual number of test samples
    num_example = num_step * BATCH_SIZE
    print("num_step: ", num_step, "num_example:", num_example)

    # restore model
    saver = tf.train.import_meta_graph(r"E:\CNN\AM-ResNet34\fine_parameters-28.meta")
    graph = tf.get_default_graph()
    # retrieve tensors, operations, ect
    image = graph.get_tensor_by_name("image_placeholder:0")
    label = graph.get_tensor_by_name("label_placeholder:0")
    output = graph.get_tensor_by_name("net/AM-ResNet34/fc_1/add:0")

    #net/ResNet/fc_1/add:0
    #net/ResNet/fc_1/add:0
    #net/GoogLeNet/fc_1/add:0
    #net/VGG16/fc_3/add:0


    sess1 = tf.Session()
    saver.restore(sess1, r"E:\CNN\AM-ResNet34\fine_parameters-28")
    sess2 = tf.Session()
    saver.restore(sess2, r"E:\CNN\AM-ResNet34\fine_parameters-29")
    sess3 = tf.Session()
    saver.restore(sess3, r"E:\CNN\AM-ResNet34\fine_parameters-30")



    coord = tf.train.Coordinator()
    threads_1 = tf.train.start_queue_runners(sess=sess1, coord=coord)
    threads_2 = tf.train.start_queue_runners(sess=sess2, coord=coord)
    threads_3 = tf.train.start_queue_runners(sess=sess3, coord=coord)
    output_set_1 = []
    output_set_2 = []
    output_set_3 = []
    label_set_1 = []
    label_set_2 = []
    label_set_3 = []
    output_set_1_index = []
    output_set_2_index = []
    output_set_3_index = []
    for i in range(num_step):
        image_data_1, label_data_1 = sess1.run([ima, lab])
        #print(image_data, label_data)
        image_data_2, label_data_2 = sess1.run([ima, lab])
        image_data_3, label_data_3 = sess1.run([ima, lab])

        index_label_1 = np.argmax(label_data_1, axis=1)
        index_label_2 = np.argmax(label_data_2, axis=1)
        index_label_3 = np.argmax(label_data_3, axis=1)
        label_set_1.append(index_label_1)
        label_set_2.append(index_label_2)
        label_set_3.append(index_label_3)

        pre_1 = sess1.run(tf.nn.softmax(output), feed_dict={image:image_data_1, label:label_data_1})
        pre_2 = sess2.run(tf.nn.softmax(output), feed_dict={image: image_data_2, label: label_data_2})
        pre_3 = sess3.run(tf.nn.softmax(output), feed_dict={image: image_data_3, label: label_data_3})
        #print(pre)
        index_pre_1 = np.argmax(pre_1, axis=1)
        index_pre_2 = np.argmax(pre_2, axis=1)
        index_pre_3 = np.argmax(pre_3, axis=1)
        output_set_1_index.append(index_pre_1)
        output_set_2_index.append(index_pre_2)
        output_set_3_index.append(index_pre_3)
        for j in range(BATCH_SIZE):
            output_set_1.append(pre_1[j][index_pre_1[j]])
        for k in range(BATCH_SIZE):
            output_set_2.append(pre_1[k][index_pre_2[k]])
        for m in range(BATCH_SIZE):
            output_set_3.append(pre_3[m][index_pre_3[m]])

    output_set_1 = np.reshape(np.array(output_set_1), num_example)
    output_set_2 = np.reshape(np.array(output_set_2), num_example)
    output_set_3 = np.reshape(np.array(output_set_3), num_example)
    label_set_1 = np.reshape(np.array(label_set_1), num_example)
    label_set_2 = np.reshape(np.array(label_set_2), num_example)
    label_set_3 = np.reshape(np.array(label_set_3), num_example)
    output_set_1_index = np.reshape(np.array(output_set_1_index), num_example)
    output_set_2_index = np.reshape(np.array(output_set_2_index), num_example)
    output_set_3_index = np.reshape(np.array(output_set_3_index), num_example)

    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    FPR_1, TPR_1, thresholds_1 = roc_curve(label_set_1, output_set_1)
    FPR_2, TPR_2, thresholds_2 = roc_curve(label_set_2, output_set_2)
    FPR_3, TPR_3, thresholds_3 = roc_curve(label_set_3, output_set_3)
    AUC_1 = auc(FPR_1, TPR_1)
    AUC_2 = auc(FPR_2, TPR_2)
    AUC_3 = auc(FPR_3, TPR_3)
    plt.figure(figsize=(8,6))
    plt.plot(FPR_1[:], TPR_1[:], "b", linewidth=2.0, label="ROC curve_epoch_28(area=%0.5f)" % AUC_1)
    plt.plot(FPR_2[:], TPR_2[:], "r", linewidth=2.0, label="ROC curve_epoch_29(area=%0.5f)" % AUC_2)
    plt.plot(FPR_3[:], TPR_3[:], "y", linewidth=2.0, label="ROC curve_epoch_30(area=%0.5f)" % AUC_3)
    plt.plot((0, 1), (0, 1), linestyle="--", linewidth=2.0)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title("Roc Curve", fontsize=18)
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    #plt.grid(True)
    plt.legend()  # display label
    plt.show()

    #https://scikit-learn.org/dev/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    f = open(r"E:\CNN\valid(AM-ResNet34)", "w")
    precision_1, recall_1, fbeta_score_1, support_1 = precision_recall_fscore_support(label_set_1, output_set_1_index, average="macro")
    precision_2, recall_2, fbeta_score_2, support_2 = precision_recall_fscore_support(label_set_2, output_set_2_index, average="macro")
    precision_3, recall_3, fbeta_score_3, support_3 = precision_recall_fscore_support(label_set_3, output_set_3_index, average="macro")
    f.write("evaluation results on validation data:"+"\n"+
            "Total test examples: {}".format(num_example)+"\n"+
            "Precision_1: {:.3f}%".format(100*precision_1) + "\n" +
            "Recall_1: {:.3f}%".format(100*recall_1)+"\n"+
            "F1_1: {:.3f}%".format(100*fbeta_score_1)+"\n"+
            "Precision_2: {:.3f}%".format(100 * precision_2) + "\n" +
            "Recall_2: {:.3f}%".format(100 * recall_2) + "\n" +
            "F1_2: {:.3f}%".format(100 * fbeta_score_2)+ "\n" +
            "Precision_3: {:.3f}%".format(100 * precision_3) + "\n" +
            "Recall_3: {:.3f}%".format(100 * recall_3) + "\n" +
            "F1_3: {:.3f}%".format(100 * fbeta_score_3))
    f.close()
    coord.request_stop()
    coord.join(threads_1)
    coord.join(threads_2)
    coord.join(threads_3)
    sess1.close()
    sess2.close()
    sess3.close()

if __name__=="__main__":
    test()
