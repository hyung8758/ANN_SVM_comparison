% 2016, spring semester team project. 
% toLibsvmForm
% 
%                                                             Hyungwon Yang
%                                                                2016.06.10
%                                                                 EMCS labs

% Run this script in order to generate train datasets.
% Run libsvm_train.sh in terminal or linux command line.
% Generating and training the datasets will take amount of time.
% I recommand you to run this work on your 2nd computer server.

% Transform mat file to libsvm form
% MNIST
train_num = 60000;
test_num = 10000;

load mnist_input
load mnist_target
load mnist_hog_input
load mnist_zca_input

input = {'mnist_input','mnist_hog_input','mnist_zca_input'};
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

load cifar10_input
load cifar10_target
load cifar10_gray_input
load cifar10_hog_input
load cifar10_zca_input

input = {'cifar10_input','cifar10_gray_input','cifar10_hog_input','cifar10_zca_input'};
target = cifar10_target;

for data = 1:length(input)
   
    tmp = eval(input{data});
    mat2libsvm(tmp(1:train_num,:),target(1:train_num,:),[input{data} '_train'])
    mat2libsvm(tmp(train_num+1:train_num+test_num,:),target(train_num+1:train_num+test_num,:),[input{data} '_test'])
end