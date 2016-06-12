% 2016, spring semester team project. 
% toLibsvmForm
% 
%                                                             Hyungwon Yang
%                                                                2016.06.10
%                                                                 EMCS labs



% Transform mat file to libsvm form
% MNIST
train_num = 60000;
test_num = 10000;

load MNIST_input
load MNIST_target
load MNIST_hog_input
load MNIST_zca_input

input = {'MNIST_input','MNIST_hog_input','MNIST_zca_input'};
target = MNIST_target;

for data = 1:length(input)
   
    tmp = eval(input{data});
    mat2libsvm(tmp(1:train_num,:),target(1:train_num,:),[input{data} '_train'])
    mat2libsvm(tmp(train_num+1:train_num+test_num,:),target(train_num+1:train_num+test_num,:),[input{data} '_test'])
end
%% CIFAR10
clear; clc; close all;
train_num = 50000;
test_num = 10000;

load CIFAR10_input
load CIFAR10_target
load CIFAR10_gray_input
load CIFAR10_hog_input
load CIFAR10_zca_input

input = {'cifar10_input','cifar10_gray_input','cifar10_hog_input','cifar10_zca_input'};
target = cifar10_target;

for data = 1:length(input)
   
    tmp = eval(input{data});
    mat2libsvm(tmp(1:train_num,:),target(1:train_num,:),[input{data} '_train'])
    mat2libsvm(tmp(train_num+1:train_num+test_num,:),target(train_num+1:train_num+test_num,:),[input{data} '_test'])
end