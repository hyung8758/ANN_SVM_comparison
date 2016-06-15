% 2016, spring semester team project. 
% extracting_features
% 
%                                                             Hyungwon Yang
%                                                                2016.05.22
%                                                                 EMCS labs

% Feature extraction of training and testing datasets.
% Two datasets are considered to be trained and the features of each 
% dataset will be extracted with a few algorithms.
% 1. MNIST dataset
%   - Original: Gray scaled and untouched original dataset.
%   - HOG     : HOG feature dataset. 
%   - ZCA     : ZCA normalized feature dataset.
% 2. CIFAR 10 dataset
%   - Original: 3 dimensional (RGB) color map dataset.
%   - gray    : Gray-scaled dataset.
%   - HOG     : HOG feature dataset. 
%   - ZCA     : ZCA normalized feature dataset.


%% Direction
% 1. Load the original dataset.
% 2. Extract features from the original datasets.
function extracting_features()

%% %%%%%%%%%%%%%%%%%%%%
%       MNIST         %
%%%%%%%%%%%%%%%%%%%%%%%
path = pwd;
directory = dir(path);
allNames = { directory.name };
checkbox = 0;
for check = 1:length(allNames)
    if strcmp(allNames{check},'data')
        checkbox = checkbox+1;
    end
end
if checkbox == 0
    error('Please run the download_data script first.')
end

% Loading dataset.
cd data
fprintf('\n##########   MNIST   ##########\n\n')
fprintf('MNIST datasets are imported.\n')
load MNIST_input

%% 1. HOG feature dataset.
% Pre_assignment.
fprintf('Start up: hog feature extraction...\n')
var_name = 'mnist_input';
hog_vector_length = 144;
squre_length = sqrt(size(eval(var_name),2));

mnist_hog_input = zeros(length(eval(var_name)),hog_vector_length);


% Reshape and extract the feature.
fprintf('   Extracing hog features...\n')
tmp = eval(var_name);
for turn = 1:length(tmp)
    shape_data = reshape(tmp(turn,:),squre_length,squre_length);
    mnist_hog_input(turn,:) = hog_feature_vector(shape_data);
end
tmp = [];

% Save it as variables.
save('MNIST_hog_input','mnist_hog_input')
clear pre_box tmp
fprintf('   Process complete.\n')

%% 2. ZCA normalized dataset.

% Extract normalized features and save them as variables.
fprintf('Start up: zca feature extraction...\n')
fprintf('   Extracing zca...\n')
mnist_zca_input = whiten(mnist_input);

% Save it as variables.
save('MNIST_zca_input','mnist_zca_input')
fprintf('   Process complete.\n')
fprintf('### MNIST dataset feature extraction has been finished.### \n\n')


%%
%%%%%%%%%%%%%%%%%%%%%%%
%      CIFAR-10       %
%%%%%%%%%%%%%%%%%%%%%%%
clear; close all;

% Loading dataset.
fprintf('\n##########   CIFAR10   ##########\n\n')
fprintf('CIFAR10 datasets are imported.\n')
load CIFAR10_input

%% 1. Gray-scaled dataset.

% Pre-assignment.
fprintf('Start up: gray-scaling...\n')
var_name = 'cifar10_input';
gray_vector_length = 1024;
squre_length = 32;

cifar10_gray_input = zeros(length(eval(var_name)),gray_vector_length);

% Gray-scaling.
fprintf('   Gray-scaling...\n')
tmp = eval(var_name);
    
for img = 1:length(tmp)
    imgbox = reshape(tmp(img,:),squre_length,squre_length,3);
    graybox = rgb2gray(imgbox);
    cifar10_gray_input(img,:) = reshape(graybox,1,1024);
end
clear tmp

% Save it as variables.
save('CIFAR10_gray_input','cifar10_gray_input')
fprintf('Gray-scaled CIFAR10 datasets are saved.\n')
fprintf('   Process complete.\n')

%% 2. HOG feature dataset.

% Pre_assignment.
fprintf('Start up: hog feature extraction...\n')
var_name = 'cifar10_gray_input';
hog_vector_length = 324;

cifar10_hog_input = zeros(length(eval(var_name)),hog_vector_length);

% Reshape and extract the feature.
fprintf('   Extracing hog features...\n')
tmp = eval(var_name);

for turn = 1:length(tmp)
    shape_data = reshape(tmp(turn,:),32,32);
    cifar10_hog_input(turn,:) = hog_feature_vector(shape_data);
end
clear tmp

% Save it as variables.
save('CIFAR10_hog_input','cifar10_hog_input')
fprintf('   Process complete.\n')

%% 3. ZCA normalized dataset.

% Extract normalized features and save them as variables.
fprintf('Start up: zca feature extraction...\n')
fprintf('   Extracing zca...\n')
cifar10_zca_input = whiten(cifar10_gray_input);

save('CIFAR10_zca_input','cifar10_zca_input')
fprintf('   Process complete.\n')
fprintf('### CIFAR10 dataset feature extraction has been finished. ###\n\n')

cd ../
addpath('data')
