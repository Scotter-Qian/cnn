# Different Algorithms of Convolutional neural network for Identification of Tunnel Seepage

STeps:
1.You can use data_augmentation to expand the datasets.

2.Making the dataset into tf.record file by data_read.py.

3.Training model according to the file of train.py or GoogleNet21.py, ResNet34.py, AM-ResNet34.py, VGGNets16.py, VGGNets16+CBAM.py, VGG11+CBAM.py etc. Choosing property model.

4.Testing and selecting the proper model by running ROC_CURVES.py or selected_different_step.py, getting the Evaluation index,  ROC curves. 

Requires:
Tensoflow version >= 1.12
python >= 3.6
GPU: 1080Ti

command：
python *.py
