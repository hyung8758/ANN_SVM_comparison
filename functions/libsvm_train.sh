#!/bin/bash
# 2016, spring semester team project.
# matlab_ann
#
#                                                             Hyungwon Yang
#                                                                2016.06.10
#                                                                 EMCS labs


# libSVM data training.
# for linux and mac osx 

# requirements
# Training and testing datasets. (it should follow the libsvm training form)
# Please run this script in which you want to save the results.


# Only 1,3,5 models will be used.
#
# place: home/hyung8758/libsvm  [/model/model_3]
# MNIST hog
# 1. 
cd model/model_1
nohup ../../svm-train -t 2 ../data/MNIST_hog_input_train.txt 1st_mnist_hog.model &

# 3.
cd ../model_3
nohup ../../svm-train -t 2 -s 1 ../data/MNIST_hog_input_train.txt 3rd_mnist_hog.model &

# 5.
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

