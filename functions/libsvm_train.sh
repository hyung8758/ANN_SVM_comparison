#!/bin/bash
# 2016, spring semester team project.
# matlab_ann
#
#                                                             Hyungwon Yang
#                                                                2016.06.10
#                                                                 EMCS labs


# libSVM data training.
# for linux and mac osx
# If you find difficulty to train data with SVM.
# Please run this script in Linux or Mac OSX. Please notice that it will
# take a lot of time so do not run this on your main computer.

# requirements
# Training and testing datasets. (it should follow the libsvm training form)
# Please run this script in which you want to save the results.


# Train 1 - 8 models.
#
# place: home/hyung8758/libsvm  [/model/model_3]
# Setting paths
CURRENT_PATH=/home/hyung8758/libsvm
MODEL_PATH=$CURRENT_PATH/models
DATA_PATH=$CURRENT_PATH/data

# Setting options
opt_1="-t 2"
opt_2="-t 2 -b 1"
opt_3="-t 2 -s 1"
opt_4="-t 2 -c 5 -g 0.5 -e 0.01"
opt_5="-t 1 -d 4"
opt_6="-t 1 -d 5"
opt_7="-t 1 -d 6"
opt_8="-t 1 -d 4 -b 1"

# Nevigate to model directory
cd $CURRENT_PATH
mkdir models
cd $MODEL_PATH
for ((x=1;x<=8;x++)); do
    mkdir model_$x
done

# MNIST original
echo =======================
echo Training MNIST datasets
echo =======================

for ((x=1;x<=8;x++)); do

    # Train 8 types of model
    cd $MODEL_PATH"/model_"$x
    nohup $CURRENT_PATH/svm-train $opt_$x $DATA_PATH/MNIST_input_train.txt $MODEL_PATH"/model_"$x/$x_mnist.model &




# 1.
cd $MODEL_PATH"/model_1"
nohup $CURRENT_PATH/svm-train -t 2 $DATA_PATH/MNIST_hog_input_train.txt $MODEL_PATH"/model_1/"1st_mnist_hog.model &

# 2.
cd $MODEL_PATH"/model_2"
nohup $CURRENT_PATH/svm-train -t 2 -b 1 $DATA_PATH/MNIST_hog_input_train.txt $MODEL_PATH"/model_2/"2nd_mnist_hog.model &

# 3.
cd $MODEL_PATH"/model_3"
nohup $CURRENT_PATH/svm-train -t 2 -s 1 $DATA_PATH/MNIST_hog_input_train.txt $MODEL_PATH"/model_3/"3rd_mnist_hog.model &

# 4.
cd $MODEL_PATH"/model_4"
nohup $CURRENT_PATH/svm-train -t 2 -c 2.8 -g $DATA_PATH/MNIST_hog_input_train.txt $MODEL_PATH"/model_4/"4th_mnist_hog.model &

# 5.
cd $MODEL_PATH"/model_5"
nohup $CURRENT_PATH/svm-train -t 2 $DATA_PATH/MNIST_hog_input_train.txt $MODEL_PATH"/model_5/"5th_mnist_hog.model &

# 6.
cd $MODEL_PATH"/model_6"
nohup $CURRENT_PATH/svm-train -t 2 $DATA_PATH/MNIST_hog_input_train.txt $MODEL_PATH"/model_6/"6th_mnist_hog.model &

# 7.
cd $MODEL_PATH"/model_7"
nohup $CURRENT_PATH/svm-train -t 2 $DATA_PATH/MNIST_hog_input_train.txt $MODEL_PATH"/model_7/"7th_mnist_hog.model &

# 8.
cd $MODEL_PATH"/model_8"
nohup $CURRENT_PATH/svm-train -t 2 $DATA_PATH/MNIST_hog_input_train.txt $MODEL_PATH"/model_8/"8th_mnist_hog.model &


# MNIST hog
# 1. 
cd model/model_1
nohup ../../svm-train -t 2 ../data/MNIST_hog_input_train.txt 1st_mnist_hog.model &

# 2.
cd ../model_3
nohup ../../svm-train -t 2 -s 1 ../data/MNIST_hog_input_train.txt 3rd_mnist_hog.model &

# 3.
cd ../model_3
nohup ../../svm-train -t 2 -s 1 ../data/MNIST_hog_input_train.txt 3rd_mnist_hog.model &

# 4.
cd ../model_3
nohup ../../svm-train -t 2 -s 1 ../data/MNIST_hog_input_train.txt 3rd_mnist_hog.model &

# 5.
cd ../model_5
nohup ../../svm-train -t 1 -d 4 ../data/MNIST_hog_input_train.txt 5th_mnist_hog.model &

# 6.
cd ../model_5
nohup ../../svm-train -t 1 -d 4 ../data/MNIST_hog_input_train.txt 5th_mnist_hog.model &

# 7.
cd ../model_5
nohup ../../svm-train -t 1 -d 4 ../data/MNIST_hog_input_train.txt 5th_mnist_hog.model &

# 8.
cd ../model_5
nohup ../../svm-train -t 1 -d 4 ../data/MNIST_hog_input_train.txt 5th_mnist_hog.model &


# MNIST zca
# 1.
cd model/model_1
nohup ../../svm-train -t 2 ../data/MNIST_zca_input_train.txt 1st_mnist_zca.model &

# 3
cd ../model_3
nohup ../../svm-train -t 2 -s 1 ../data/MNIST_zca_input_train.txt 3rd_mnist_zca.model &

# 5
cd ../model_5
nohup ../../svm-train -t 1 -d 4 ../data/MNIST_zca_input_train.txt 5th_mnist_zca.model &

##################################
# CIFAR10 original
# 1.
cd model/model_1
nohup ../../svm-train -t 2 ../data/cifar10_input_train.txt 1st_cifar10.model &

# 3.
cd ../model_3
nohup ../../svm-train -t 2 -s 1 ../data/cifar10_input_train.txt 3rd_cifar10.model &

# 5.
cd ../model_5
nohup ../../svm-train -t 1 -d 4 ../data/cifar10_input_train.txt 5th_cifar10.model &

# CIFAR10 gray-scaled
# 1.
cd model/model_1
nohup ../../svm-train -t 2 ../data/cifar10_gray_input_train.txt 1st_cifar10_gray.model &

# 3.
cd ../model_3
nohup ../../svm-train -t 2 -s 1 ../data/cifar10_gray_input_train.txt 3rd_cifar10_gray.model &

# 5.
cd ../model_5
nohup ../../svm-train -t 1 -d 4 ../data/cifar10_gray_input_train.txt 5th_cifar10_gray.model &

# CIFAR10 hog
# 1.
cd model/model_1
nohup ../../svm-train -t 2 ../data/cifar10_hog_input_train.txt 1st_cifar10_hog.model &

# 3.
cd ../model_3
nohup ../../svm-train -t 2 -s 1 ../data/cifar10_hog_input_train.txt 3rd_cifar10_hog.model &

# 5.
cd ../model_5
nohup ../../svm-train -t 1 -d 4 ../data/cifar10_hog_input_train.txt 5th_cifar10_hog.model &

# CIFAR10 zca
# 1.
cd model/model_1
nohup ../../svm-train -t 2 ../data/cifar10_zca_input_train.txt 1st_cifar10_zca.model &

# 3.
cd ../model_3
nohup ../../svm-train -t 2 -s 1 ../data/cifar10_zca_input_train.txt 3rd_cifar10_zca.model &

# 5.
cd ../model_5
nohup ../../svm-train -t 1 -d 4 ../data/cifar10_zca_input_train.txt 5th_cifar10_zca.model &

