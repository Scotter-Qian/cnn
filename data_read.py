# make tf.record file
# changing the path when running
import os
import pickle
import tensorflow as tf
import numpy as np
import math

# Define the function to search the data path to be trained
def search_file(path):
    file = list()
    for name in os.listdir(path):
        name_path = os.path.join(path, name)
        if os.path.isfile(name_path):
            if name_path.endswith(".xy"):
                file.append(name_path)
    return np.array(file)

def encode_to_tfrecords(path):
    # Create an object for recording files to write records; 
    # change "valid.tfrecord" to "train.tfrecord" when training
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    data_path_file = search_file(path)
    for file_path in data_path_file:
        file = open(file_path, "rb")
        data = pickle.load(file)
        image = (data["x"])
        image = np.reshape(image, (288, 384, 1))
        label = (data["y"])
        label = np.reshape(label, 1)
        # transfer image data to bytes
        image_raw = image.tobytes()
        #label_raw = label.tobytes()
        # Organize the data into the data structure required by TFRecord
        example = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
        # Serialize write file
        writer.write(example.SerializeToString())
    writer.close()
    print("tfrecord_file is done!")

# Read data from TFRecord, get the batch_size pictures and the corresponding label
def read_example(filename, batch_size):
    # Create a file reader to read data from the queue file
    reader = tf.TFRecordReader()
    # Create a file name queue
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    _, serialized_example = reader.read(filename_queue)
    # Read batch_size pictures each time
    min_after_dequeue = 10
    batch = tf.train.shuffle_batch([serialized_example],
                            batch_size=batch_size,
                            capacity=min_after_dequeue+2000+3 * batch_size,
                            min_after_dequeue=10,
                            num_threads=2)
    parsed_example = tf.parse_example(batch, features={'image': tf.FixedLenFeature([], tf.string),
                                                       'label': tf.FixedLenFeature([], tf.int64)})
    image_raw = tf.decode_raw(parsed_example['image'], tf.uint8)
    #IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH equals to 288, 384 and 1
    image = tf.cast(tf.reshape(image_raw, [batch_size, 288, 384, 1]), tf.float32)
    image = image/255.0
    label_raw = tf.cast(parsed_example['label'], tf.int32)
    label = tf.reshape(label_raw, [batch_size*1])
    #depth equals to num_classes
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
    # Create a coordinator to manage threads
    coord = tf.train.Coordinator()
    # Start the thread that fills the queue, and the file name starts to enter the queue at this time
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for j in range(epoch):
        for i in range(batch_nums):
            image, label = sess.run([ima, lab])
            print(label)
    coord.request_stop()
    coord.join(threads)
    sess.close()
    """
