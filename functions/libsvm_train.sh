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

# Setting 8 options.
options="-t 2,\
-t 2 -b 1,\
-t 2 -s 1,\
-t 2 -c 5 -g 0.5 -e 0.01,\
-t 1 -d 4,\
-t 1 -d 5,\
-t 1 -d 6,\
-t 1 -d 4 -b 1"

# Nevigate to model directory.
cd $CURRENT_PATH
mkdir models
cd $MODEL_PATH

# make model directories.
for ((x=1;x<=8;x++)); do
    mkdir model_$x
done

for x in {1..8};do
echo ===========
tmp_var=$(echo $options | sed -n 1'p' | tr ',' '\n'| sed -n ${x}p)
done


# MNIST original.
echo =======================
echo  Running SVM models...
echo =======================

for ((x=1;x<=8;x++)); do

    # Train 8 types of model.
    cd $MODEL_PATH"/model_"$x
    tmp_var=$(echo $options | sed -n 1'p' | tr ',' '\n'| sed -n ${x}p)

    # MNIST_origianl
    nohup $CURRENT_PATH/svm-train tmp_var $DATA_PATH/mnist_input_train.txt $MODEL_PATH"/model_"$x/$x"_mnist.model" &

    # MNIST_hog
    nohup $CURRENT_PATH/svm-train tmp_var $DATA_PATH/mnist_hog_input_train.txt $MODEL_PATH"/model_"$x/$x"_mnist_hog.model" &

    # MNIST_zca
    nohup $CURRENT_PATH/svm-train tmp_var $DATA_PATH/mnist_zca_input_train.txt $MODEL_PATH"/model_"$x/$x"_mnist_zca.model" &

    # CIFAR10_origianl
    nohup $CURRENT_PATH/svm-train tmp_var $DATA_PATH/cifar10_input_train.txt $MODEL_PATH"/model_"$x/$x"_cifar10.model" &

    # CIFAR10_gray
    nohup $CURRENT_PATH/svm-train tmp_var $DATA_PATH/cifar10_gray_input_train.txt $MODEL_PATH"/model_"$x/$x"_cifar10_gray.model" &

    # CIFAR10_hog
    nohup $CURRENT_PATH/svm-train tmp_var $DATA_PATH/cifar10_hog_input_train.txt $MODEL_PATH"/model_"$x/$x"_cifar10_hog.model" &

    # CIFAR10_zca
    nohup $CURRENT_PATH/svm-train tmp_var $DATA_PATH/cifar10_zca_input_train.txt $MODEL_PATH"/model_"$x/$x"_cifar10_zca.model" &
done

echo ====================================================================
echo  Setting up finished... Please wait until all the training is over.
echo ====================================================================
