% 2016, spring semester team project. 
% matlab_ann
% 
%                                                             Hyungwon Yang
%                                                                2016.06.10
%                                                                 EMCS labs



%% MNIST training.

% Imporint data
fprintf('Importing MNIST data\n')
load MNIST_input
load MNIST_target
load MNIST_hog_input
load MNIST_zca_input

% Normalizing datasets.
%MNIST_input = zscore(MNIST_input);
MNIST_hog_input = zscore(MNIST_hog_input);
MNIST_zca_input = zscore(MNIST_zca_input);

% ANN
% Parameter settings.
hid_units = 700;
hog_units = 100;
activationFunction = @logisticSigmoid;
dActivationFunction = @dLogisticSigmoid;

% Allocating datasets.
train_input = MNIST_input(1:train_num,:)';
train_hog_input = MNIST_hog_input(1:train_num,:)';
train_zca_input = MNIST_zca_input(1:train_num,:)';
mnist_labels = MNIST_target(1:train_num,:);

% Transform the labels to correct target values.
targetValues = 0.*ones(10, size(mnist_labels, 1));
for n = 1: size(mnist_labels, 1)
    targetValues(mnist_labels(n) + 1, n) = 1;
end;
% Training data
fprintf('Training 3 types of data by ANN.\n')
fprintf('Training 1st data... ')
[input_hid_w, input_out_w] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, hid_units, train_input, targetValues, mnist_epochs, mnist_batch, mnist_lr);
fprintf('finished.\n')
fprintf('Training 2nd data... ')
[hog_hid_w, hog_out_w] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, hog_units, train_hog_input, targetValues, mnist_epochs, mnist_batch, mnist_lr);
fprintf('finished.\n')
fprintf('Training 3rd data... ')
[zca_hid_w, zca_out_w] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, hid_units, train_zca_input, targetValues, mnist_epochs, mnist_batch, mnist_lr);
fprintf('finished.\n')

% Allocating validation set.
test_input = MNIST_input(train_num+1:train_num+test_num,:)';
test_hog_input = MNIST_hog_input(train_num+1:train_num+test_num,:)';
test_zca_input = MNIST_zca_input(train_num+1:train_num+test_num,:)';
mnist_labels = MNIST_target(train_num+1:train_num+test_num,:);

% Testing data.
fprintf('Testing 3 types of data by ANN\n')
fprintf('Testing 1st data... ')
[input_corcls, input_clserr] = validateTwoLayerPerceptron(activationFunction, input_hid_w, input_out_w, test_input, mnist_labels);
fprintf('finished.\n')
fprintf('Testing 2nd data... ')
[hog_corcls, hog_clserr] = validateTwoLayerPerceptron(activationFunction, hog_hid_w, hog_out_w, test_hog_input, mnist_labels);
fprintf('finished.\n')
fprintf('Testing 3rd data... ')
[zca_corcls, zca_clserr] = validateTwoLayerPerceptron(activationFunction, zca_hid_w, zca_out_w, test_zca_input, mnist_labels);
fprintf('finished.\n')

% Save the results.
input_accuracy = input_corcls/length(mnist_labels)*100;
hog_accuracy = hog_corcls/length(mnist_labels)*100;
zca_accuracy = zca_corcls/length(mnist_labels)*100;

%save('MNIST_ANN_RESULT','input_accuracy','hog_accuracy','zca_accuracy')
fprintf('##### MNIST ANN result was saved successfully. #####\n')
close all;

%% CIFAR10 training.
% Parameter resetting.
train_num = 50000; 
test_num = 10000; 
% lr: learning rate, batch = batch size.
% The optimal number of hidden units are set. However, adjusting the number
% is possible. Please refer to the related variables below.
mnist_lr = 0.1;
mnist_batch = 100;
mnist_epochs = 300;

cifar10_lr = 0.01;
cifar10_batch = 100;
cifar10_epochs = 300;

% Imporint data
fprintf('Importing CIFAR10 data\n')
load CIFAR10_input
load CIFAR10_target
load CIFAR10_gray_input
load CIFAR10_hog_input
load CIFAR10_zca_input

% Normalizing datasets.
cifar10_input = zscore(cifar10_input);
cifar10_gray_input = zscore(cifar10_gray_input);
cifar10_hog_input = zscore(cifar10_hog_input);
cifar10_zca_input = zscore(cifar10_zca_input);


% ANN
% Parameter settings.
origin_units = 500; % 2000
hid_units = 400; % 800
hog_units = 200; % 300
activationFunction = @logisticSigmoid;
dActivationFunction = @dLogisticSigmoid;

% Allocating datasets.
train_input = cifar10_input(1:train_num,:)';
train_gray_input = cifar10_gray_input(1:train_num,:)';
train_hog_input = cifar10_hog_input(1:train_num,:)';
train_zca_input = cifar10_zca_input(1:train_num,:)';
cifar10_labels = cifar10_target(1:train_num,:);

% Transform the labels to correct target values.
targetValues = 0.*ones(10, size(cifar10_labels, 1));
for n = 1: size(cifar10_labels, 1)
    targetValues(cifar10_labels(n) + 1, n) = 1;
end;

% Training data
fprintf('Training 4 types of data by ANN.\n')
fprintf('Training 1st data... ')
[input_hid_w, input_out_w] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, origin_units, train_input, targetValues, cifar10_epochs, cifar10_batch, cifar10_lr);
fprintf('finished.\n')
fprintf('Training 2nd data... ')
[gray_hid_w, gray_out_w] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, hid_units, train_gray_input, targetValues, cifar10_epochs, cifar10_batch, cifar10_lr);
fprintf('finished.\n')
fprintf('Training 3rd data... ')
[hog_hid_w, hog_out_w] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, hog_units, train_hog_input, targetValues, cifar10_epochs, cifar10_batch, cifar10_lr);
fprintf('finished.\n')
fprintf('Training 4th data... ')
[zca_hid_w, zca_out_w] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, hid_units, train_zca_input, targetValues, cifar10_epochs, cifar10_batch, cifar10_lr);
fprintf('finished.\n')

% Allocating validation set.
test_input = cifar10_input(train_num+1:train_num+test_num,:)';
test_gray_input = cifar10_gray_input(train_num+1:train_num+test_num,:)';
test_hog_input = cifar10_hog_input(train_num+1:train_num+test_num,:)';
test_zca_input = cifar10_zca_input(train_num+1:train_num+test_num,:)';
cifar10_labels = cifar10_target(train_num+1:train_num+test_num,:);

% Testing data.
fprintf('Testing 4 types of data by ANN.\n')
fprintf('Testing 1st data... ')
[input_corcls, ~] = validateTwoLayerPerceptron(activationFunction, input_hid_w, input_out_w, test_input, cifar10_labels);
fprintf('finished.\n')
fprintf('Testing 2nd data... ')
[gray_corcls, ~] = validateTwoLayerPerceptron(activationFunction, gray_hid_w, gray_out_w, test_gray_input, cifar10_labels);
fprintf('finished.\n')
fprintf('Testing 3rd data... ')
[hog_corcls, ~] = validateTwoLayerPerceptron(activationFunction, hog_hid_w, hog_out_w, test_hog_input, cifar10_labels);
fprintf('finished.\n')
fprintf('Testing 4th data... ')
[zca_corcls, ~] = validateTwoLayerPerceptron(activationFunction, zca_hid_w, zca_out_w, test_zca_input, cifar10_labels);
fprintf('finished.\n')

% Save the results.
input_accuracy = input_corcls/length(cifar10_labels)*100;
gray_accuracy = gray_corcls/length(cifar10_labels)*100;
hog_accuracy = hog_corcls/length(cifar10_labels)*100;
zca_accuracy = zca_corcls/length(cifar10_labels)*100;

save('CIFAR10_ANN_RESULT','input_accuracy','gray_accuracy','hog_accuracy','zca_accuracy')
fprintf('##### CIFAR10 ANN result was saved successfully. #####\n')
close all;