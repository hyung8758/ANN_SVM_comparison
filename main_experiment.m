% 2016 Pattern Recognition Term Project 
%
%                                                             Hyungwon Yang
%                                                               SungSoo kim
%                                                                2016.06.05
%
% OS compatibility test:
% Tested successfully:
% - Mac OSX, Windows
% Not tested: 
% - Linux
%
% Error Reports:
% 1. This script is tested on Matlab ver.R2016a. libsvm compiling error 
%    is reported on R2015b.
% 2. If gunzip related error occurred, please restart the download_data 
%    function. Error can be occurred when the data has not been 
%    downloaded completely. Check your internet connection too.
% 3. When you run the sections of this script, make sure that your current
%    path is located in ~/project_script. If you run the sections when the
%    current path is located in the other folders such as 'data', 
%    or 'functions', the script might give you an error. 
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (1) Data Preparation %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;
addpath(genpath(pwd))

% Loading Data.
% Allocating MNIST and CIFAR10 dataset in a 'data' folder.
% Its processing time will vary depending on the speed of the internet server.
download_data()
fprintf('##### All data was downloaded successfully. #####\n')

% Extracting Data Features.
extracting_features()
fprintf('##### All features were extracted. #####\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (2) ANN and SVM algorithms Preparation %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Downloading ANN and SVM scripts.
% ANN
% Our team used to apply this package. Due to the low performance, however,
% we are not going to use this package anymore but train the datasets
% with nnstart (matlab built in package) instead. 
% If anyone would like to try this package, feel free to use it.

% download_ANN()
% fprintf('##### ANN was downloaded successfully. #####\n')

% SVM
download_SVM()
fprintf('##### SVM was downloaded successfully. #####\n')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (3) Training Data (Recipe) %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the numbers of training and testing datasets.
% Using Full datasets will take a lot of time especially in SVM training
% session. Please adjust the numbers of the datasets (For testing purpose,
% set the training and testing numbers small as possible.)
% Those parameters are applied to all learning algorithms.

% MNIST training.

% Importing data.
fprintf('Importing MNIST data\n')
load MNIST_input
load MNIST_target
load MNIST_hog_input
load MNIST_zca_input

% Data resetting for training.
mnist_input = mnist_input';
mnist_hog_input = mnist_hog_input';
mnist_zca_input = mnist_zca_input';
mnist_target = spreadTarget(mnist_target);

% ANN
% Setting hyper parameters
fprintf('Organizing data and parameters.\n')
% For more information of nnstart please refer to the homepage below
% > http://www.mathworks.com/help/nnet/functionlist-alpha.html

% The number of hidden units
hiddenUnits = 300;
hiddenUnits_hog = 100;
hiddenUnits_zca = 300;

% Scaled conjugate gradient backpropagation.
trainFcn = 'trainscg'; 
performFcn = 'crossentropy';
mnist_net = patternnet(hiddenUnits, trainFcn, performFcn);
mnist_hog_net = patternnet(hiddenUnits_hog, trainFcn, performFcn);
mnist_zca_net = patternnet(hiddenUnits_zca, trainFcn, performFcn);

% Set the data proportion of train, test, and validation.
mnist_net.divideParam.trainRatio = 70/100;
mnist_net.divideParam.valRatio = 15/100;
mnist_net.divideParam.testRatio = 15/100;

mnist_hog_net.divideParam.trainRatio = 70/100;
mnist_hog_net.divideParam.valRatio = 15/100;
mnist_hog_net.divideParam.testRatio = 15/100;

mnist_zca_net.divideParam.trainRatio = 70/100;
mnist_zca_net.divideParam.valRatio = 15/100;
mnist_zca_net.divideParam.testRatio = 15/100;

% Set the activation functions for hidden and output layers
mnist_net.layers{1}.transferFcn = 'tansig';
mnist_net.layers{2}.transferFcn = 'softmax';

mnist_hog_net.layers{1}.transferFcn = 'tansig';
mnist_hog_net.layers{2}.transferFcn = 'softmax';

mnist_zca_net.layers{1}.transferFcn = 'tansig';
mnist_zca_net.layers{2}.transferFcn = 'softmax';

% Training data. (validation and testing is included)
fprintf('Training 3 types of data by ANN.\nTraining 1st data... ')
[mnist_net, mnist_accuracy] = ANNtrain(mnist_net,mnist_input,mnist_target);
fprintf('finished.\nTraining 2nd data... ')
[mnist_hog_net, mnist_hog_accuracy] = ANNtrain(mnist_hog_net,mnist_hog_input,mnist_target);
fprintf('finished.\nTraining 3rd data... ')
[mnist_zca_net, mnist_zca_accuracy] = ANNtrain(mnist_zca_net,mnist_zca_input,mnist_target);
fprintf('finished.\n')

% Save the results.
fprintf('Saving the results.\n')
result.mnist = {mnist_net,mnist_accuracy};
result.mnist_hog = {mnist_hog_net,mnist_hog_accuracy};
result.mnist_zca = {mnist_zca_net,mnist_zca_accuracy};

save('MNIST_ANN_result','result')
fprintf('##### MNIST ANN datasets were trained successfully. #####\n')
clear; close all;

%% SVM
% When you run the libsvm code, you need to run it under the 'matlab_svm'
% folder. Otherwise you will use build-in svmtrain function which isn't
% libsvm function.

% Imporing data.
fprintf('Importing MNIST data.\n')
load MNIST_input
load MNIST_target
load MNIST_hog_input
load MNIST_zca_input

% Parameter settings.
fprintf('Organizing data and parameters.\n')
train_num = 60000;
test_num = 10000;

% First model parameters: mnist_original
sp01.name = 'MNIST_input';
sp01.train_input = mnist_input(1:train_num,:);
sp01.train_target = mnist_target(1:train_num,:);
sp01.test_input = mnist_input(train_num+1:train_num+test_num,:);
sp01.test_target = mnist_target(train_num+1:train_num+test_num,:);
sp01.option = '-t 2';

% mnist_hog
sp11.name = 'MNIST_hog_input';
sp11.train_input = mnist_hog_input(1:train_num,:);
sp11.train_target = mnist_target(1:train_num,:);
sp11.test_input = mnist_hog_input(train_num+1:train_num+test_num,:);
sp11.test_target = mnist_target(train_num+1:train_num+test_num,:);
sp11.option = '-t 2';

% mnist_zca
sp21.name = 'MNIST_zca_iuput';
sp21.train_input = mnist_zca_input(1:train_num,:);
sp21.train_target = mnist_target(1:train_num,:);
sp21.test_input = mnist_zca_input(train_num+1:train_num+test_num,:);
sp21.test_target = mnist_target(train_num+1:train_num+test_num,:);
sp21.option = '-t 2';

% Second model parameters: mnist_original
sp02.name = 'MNIST_input';
sp02.train_input = mnist_input(1:train_num,:);
sp02.train_target = mnist_target(1:train_num,:);
sp02.test_input = mnist_input(train_num+1:train_num+test_num,:);
sp02.test_target = mnist_target(train_num+1:train_num+test_num,:);
sp02.option = '-t 1 -d 4';

% mnist_hog
sp12.name = 'MNIST_hog_input';
sp12.train_input = mnist_hog_input(1:train_num,:);
sp12.train_target = mnist_target(1:train_num,:);
sp12.test_input = mnist_hog_input(train_num+1:train_num+test_num,:);
sp12.test_target = mnist_target(train_num+1:train_num+test_num,:);
sp12.option = '-t 1 -d 4';

% mnist_zca
sp22.name = 'MNIST_zca_input';
sp22.train_input = mnist_zca_input(1:train_num,:);
sp22.train_target = mnist_target(1:train_num,:);
sp22.test_input = mnist_zca_input(train_num+1:train_num+test_num,:);
sp22.test_target = mnist_target(train_num+1:train_num+test_num,:);
sp22.option = '-t 1 -d 4';

% 1st recipe: option(-t 2)
% Use Radial Basis Function.
% Set other parameters as default.
fprintf('1st svm model training...\nTraining 1st model 1st data... ')
mnist_accuracy1 = run_svm(sp01);
fprintf('finished.\nTraining 1st model 2nd data... ')
mnist_hog_accuracy1 = run_svm(sp11);
fprintf('finished.\nTraining 1st model 3rd data... ')
mnist_zca_accuracy1 = run_svm(sp21);
fprintf('finished.\n')

% 2nd recipe: option(-t 1 -d 4)
% Use 4th degrees of polynomial function.
% Set other parameters as default.
fprintf('2nd svm model training...\nTraining 2nd model 1st data... ')
mnist_accuracy2 = run_svm(sp02);
fprintf('finished.\nTraining 2nd model 2nd data... ')
mnist_hog_accuracy2 = run_svm(sp12);
fprintf('finished.\nTraining 2nd model 3rd data... ')
mnist_zca_accuracy2 = run_svm(sp22);
fprintf('finished.\n')

% Save the results.
fprintf('Saving the results.\n')
result.mnist = {mnist_accuracy1, mnist_accuracy2};
result.mnist_hog = {mnist_hog_accuracy1, mnist_hog_accuracy2};
result.mnist_zca = {mnist_zca_accuracy1; mnist_zca_accuracy2};

save('MNIST_SVM_result','result')
fprintf('##### MNIST SVM datasets were trained successfully. #####\n')
clear; close all;

%% CIFAR10 training.

% Importing data.
fprintf('Importing CIFAR10 data\n')
load CIFAR10_input
load CIFAR10_target
load CIFAR10_gray_input
load CIFAR10_hog_input
load CIFAR10_zca_input

% Data resetting for training.
fprintf('Organizing data and parameters.\n')
cifar10_input = cifar10_input';
cifar10_gray_input = cifar10_gray_input';
cifar10_hog_input = cifar10_hog_input';
cifar10_zca_input = cifar10_zca_input';
cifar10_target = spreadTarget(cifar10_target);

% ANN
% Setting hyper parameters
% For more information of nnstart please refer to the homepage below
% > http://www.mathworks.com/help/nnet/functionlist-alpha.html

% The number of hidden units
hiddenUnits = 1000;
hiddenUnits_gray = 700;
hiddenUnits_hog = 200;
hiddenUnits_zca = 700;

% Scaled conjugate gradient backpropagation
trainFcn = 'trainscg'; 
performFcn = 'crossentropy';
cifar10_net = patternnet(hiddenUnits, trainFcn, performFcn);
cifar10_gray_net = patternnet(hiddenUnits_gray, trainFcn, performFcn);
cifar10_hog_net = patternnet(hiddenUnits_hog, trainFcn, performFcn);
cifar10_zca_net = patternnet(hiddenUnits_zca, trainFcn, performFcn);

% Set the data proportion of train, test, and validation.
cifar10_net.divideParam.trainRatio = 70/100;
cifar10_net.divideParam.valRatio = 15/100;
cifar10_net.divideParam.testRatio = 15/100;

cifar10_gray_net.divideParam.trainRatio = 70/100;
cifar10_gray_net.divideParam.valRatio = 15/100;
cifar10_gray_net.divideParam.testRatio = 15/100;

cifar10_hog_net.divideParam.trainRatio = 70/100;
cifar10_hog_net.divideParam.valRatio = 15/100;
cifar10_hog_net.divideParam.testRatio = 15/100;

cifar10_zca_net.divideParam.trainRatio = 70/100;
cifar10_zca_net.divideParam.valRatio = 15/100;
cifar10_zca_net.divideParam.testRatio = 15/100;

% Set the activation functions for hidden and output layers
cifar10_net.layers{1}.transferFcn = 'tansig';
cifar10_hog_net.layers{2}.transferFcn = 'softmax';

cifar10_gray_net.layers{1}.transferFcn = 'tansig';
cifar10_gray_net.layers{2}.transferFcn = 'softmax';

cifar10_hog_net.layers{1}.transferFcn = 'tansig';
cifar10_hog_net.layers{2}.transferFcn = 'softmax';

cifar10_zca_net.layers{1}.transferFcn = 'tansig';
cifar10_zca_net.layers{2}.transferFcn = 'softmax';

% Training data. (validation and testing is included)
fprintf('Training 3 types of data by ANN.\nTraining 1st data... ')
[cifar10_net, cifar10_accuracy] = ANNtrain(cifar10_net,cifar10_input,cifar10_target);
fprintf('finished.\nTraining 2nd data... ')
[cifar10_gray_net, cifar10_gray_accuracy] = ANNtrain(cifar10_gray_net,cifar10_gray_input,cifar10_target);
fprintf('finished.\nTraining 3rd data... ')
[cifar10_hog_net, cifar10_hog_accuracy] = ANNtrain(cifar10_hog_net,cifar10_hog_input,cifar10_target);
fprintf('finished.\nTraining 4th data... ')
[cifar10_zca_net, cifar10_zca_accuracy] = ANNtrain(cifar10_zca_net,cifar10_zca_input,cifar10_target);
fprintf('finished.\n')

Save the results.
fprintf('Saving the results.\n')
result.cifar10 = {cifar10_net, cifar10_accuracy};
result.cifar10_gray = {cifar10_gray_net, cifar10_gray_accuracy};
result.cifar10_hog = {cifar10_hog_net, cifar10_hog_accuracy};
result.cifar10_zca = {cifar10_zca_net, cifar10_zca_accuracy};

save('CIFAR10_ANN_result','result')
fprintf('##### CIFAR10 ANN datasets were trained successfully. #####\n')
clear; close all;

%% SVM
% When you run the libsvm code, you need to run it under the 'matlab_svm'
% folder. Otherwise you will use build-in svmtrain function which isn't
% libsvm function.

% Importing data.
fprintf('Importing CIFAR10 data\n')
load CIFAR10_input
load CIFAR10_target
load CIFAR10_gray_input
load CIFAR10_hog_input
load CIFAR10_zca_input

% Parameter settings.
fprintf('Organizing data and parameters.\n')
train_num = 50000;
test_num = 10000;

% First model parameters: cifar10_original
sp01.name = 'CIFAR10_input';
sp01.train_input = cifar10_input(1:train_num,:);
sp01.train_target = cifar10_target(1:train_num,:);
sp01.test_input = cifar10_input(train_num+1:train_num+test_num,:);
sp01.test_target = cifar10_target(train_num+1:train_num+test_num,:);
sp01.option = '-t 2';

% cifar10_gray
sp11.name = 'CIFAR10_gary_input';
sp11.train_input = cifar10_gray_input(1:train_num,:);
sp11.train_target = cifar10_target(1:train_num,:);
sp11.test_input = cifar10_gray_input(train_num+1:train_num+test_num,:);
sp11.test_target = cifar10_target(train_num+1:train_num+test_num,:);
sp11.option = '-t 2';

% cifar10_hog
sp21.name = 'CIFAR10_hog_input';
sp21.train_input = cifar10_hog_input(1:train_num,:);
sp21.train_target = cifar10_target(1:train_num,:);
sp21.test_input = cifar10_hog_input(train_num+1:train_num+test_num,:);
sp21.test_target = cifar10_target(train_num+1:train_num+test_num,:);
sp21.option = '-t 2';

% cifar10_zca
sp31.name = 'CIFAR10_zca_input';
sp31.train_input = cifar10_zca_input(1:train_num,:);
sp31.train_target = cifar10_target(1:train_num,:);
sp31.test_input = cifar10_zca_input(train_num+1:train_num+test_num,:);
sp31.test_target = cifar10_target(train_num+1:train_num+test_num,:);
sp31.option = '-t 2';


% Second model parameters: cifar10_original
sp02.name = 'CIFAR10_input';
sp02.train_input = cifar10_input(1:train_num,:);
sp02.train_target = cifar10_target(1:train_num,:);
sp02.test_input = cifar10_input(train_num+1:train_num+test_num,:);
sp02.test_target = cifar10_target(train_num+1:train_num+test_num,:);
sp02.option = '-t 1 -d 4';

% cifar10_gray
sp12.name = 'CIFAR10_gary_input';
sp12.train_input = cifar10_gray_input(1:train_num,:);
sp12.train_target = cifar10_target(1:train_num,:);
sp12.test_input = cifar10_gray_input(train_num+1:train_num+test_num,:);
sp12.test_target = cifar10_target(train_num+1:train_num+test_num,:);
sp12.option = '-t 1 -d 4';

% cifar10_hog
sp22.name = 'CIFAR10_hog_input';
sp22.train_input = cifar10_hog_input(1:train_num,:);
sp22.train_target = cifar10_target(1:train_num,:);
sp22.test_input = cifar10_hog_input(train_num+1:train_num+test_num,:);
sp22.test_target = cifar10_target(train_num+1:train_num+test_num,:);
sp22.option = '-t 1 -d 4';

% cifar10_zca
sp32.name = 'CIFAR10_zca_input';
sp32.train_input = cifar10_zca_input(1:train_num,:);
sp32.train_target = cifar10_target(1:train_num,:);
sp32.test_input = cifar10_zca_input(train_num+1:train_num+test_num,:);
sp32.test_target = cifar10_target(train_num+1:train_num+test_num,:);
sp32.option = '-t 1 -d 4';


% 1st recipe: option(-t 2)
% Use Radial Basis Function.
% Set other parameters as default.
fprintf('First svm model training...\nTraining 1st model 1st data... ')
cifar10_accuracy1 = run_svm(sp01);
fprintf('finished.\nTraining 1st model 2nd data... ')
cifar10_gray_accuracy1 = run_svm(sp11);
fprintf('finished.\nTraining 1st model 3rd data... ')
cifar10_hog_accuracy1 = run_svm(sp21);
fprintf('finished.\nTraining 1st model 4th data... ')
cifar10_zca_accuracy1 = run_svm(sp31);
fprintf('finished.\n')

% 2nd recipe: option(-t 1 -d 4)
% Use 4th degrees of polynomial function.
% Set other parameters as default.
fprintf('Second svm model training...\nTraining 2nd model 1st data... ')
cifar10_accuracy2 = run_svm(sp02);
fprintf('finished.\nTraining 2nd model 2nd data... ')
cifar10_gray_accuracy2 = run_svm(sp12);
fprintf('finished.\nTraining 2nd model 3rd data... ')
cifar10_hog_accuracy2 = run_svm(sp22);
fprintf('finished.\nTraining 2nd model 4th data... ')
cifar10_zca_accuracy2 = run_svm(sp32);
fprintf('finished.\n')

% Save the results.
fprintf('Saving the results.\n')
result.cifar10 = {cifar10_accuracy1, cifar10_accuracy2};
result.cifar10_gray = {cifar10_gray_accuracy1, cifar10_gray_accuracy2};
result.cifar10_hog = {cifar10_hog_accuracy1, cifar10_hog_accuracy2};
result.cifar10_zca = {cifar10_zca_accuracy1, cifar10_zca_accuracy2};

save('CIFAR10_SVM_result','result')
fprintf('##### CIFAR10 SVM datasets were trained successfully. #####\n')
clear; close all;

% Collecting result data into a folder.
mkdir result
movefile('./*_result.mat','./result/')
addpath('./result')
fprintf('Result data is saved in a result folder.\n')

%%%%%%%%%%%%%%%%%%%%%%%
%% (4) Testing Data %%%
%%%%%%%%%%%%%%%%%%%%%%%

% MNIST
% ANN result.
load MNIST_ANN_result
fprintf('\n### MNIST ANN RESULT ###\n')
fprintf('Original data accuracy: %0.2f\n',result.mnist{2})
fprintf('HOG data accuracy: %0.2f\n',result.mnist_hog{2})
fprintf('ZCA data accuracy: %0.2f\n',result.mnist_zca{2})
clear

% SVM result.
load MNIST_SVM_result
fprintf('\n### MNIST SVM RESULT ###\n')
fprintf(['1st model: \n'...
    '  original: %0.2f\n'...
    '  hog     : %0.2f\n'...
    '  zca     : %0.2f\n'...
    '2nd model: \n'...
    '  original: %0.2f\n'...
    '  hog     : %0.2f\n'...
    '  zca     : %0.2f\n'],result.mnist{1}(1),...
    result.mnist_hog{1}(1),result.mnist_zca{1}(1),...
    result.mnist{2}(1),result.mnist_hog{2}(1),result.mnist_zca{2}(1))
clear

% CIFAR10
% ANN result.
load CIFAR10_ANN_result
fprintf('\n### CIFAR10 ANN RESULT ###\n')
fprintf('Original data accuracy: %0.2f\n',result.cifar10{2})
fprintf('Gray data accuracy: %0.2f\n',result.cifar10_gray{2})
fprintf('HOG data accuracy: %0.2f\n',result.cifar10_hog{2})
fprintf('ZCA data accuracy: %0.2f\n',result.cifar10_zca{2})
clear

% SVM result.
load CIFAR10_SVM_result
fprintf('\n### CIFAR10 SVM RESULT ###\n')
fprintf(['1st model: \n'...
    '  original: %0.2f\n'...
    '  gray    : %0.2f\n'...
    '  hog     : %0.2f\n'...
    '  zca     : %0.2f\n'...
    '2nd model: \n'...
    '  original: %0.2f\n'...
    '  gray    : %0.2f\n'...
    '  hog     : %0.2f\n'...
    '  zca     : %0.2f\n'],result.cifar10{1}(1),...
    result.cifar10_gray{1}(1),result.cifar10_hog{1}(1),...
    result.cifar10_zca{1}(1),result.cifar10{2}(1),...
    result.cifar10_gray{2}(1),result.cifar10_hog{2}(1),...
    result.cifar10_zca{2}(1))
clear
fprintf(['\n##### All the experiment procedures finished successfully. #####\n'...
         '#####                      THANK YOU                       #####\n\n'])
