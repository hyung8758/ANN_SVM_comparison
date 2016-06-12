% 2016, spring semester team project. 
% download_data
% 
%                                                             Hyungwon Yang
%                                                                2016.06.10
%                                                                 EMCS labs


function download_data()

% Make a data folder.
mkdir data
cd data

%% MNIST
% Download and save the datasets.
fprintf('Downloading MNIST datasets... (12MB)\n')
mnist_data_address = {'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'};

save_mnist{1} = websave('train-images-idx3-ubyte',mnist_data_address{1});
save_mnist{2} = websave('train-labels-idx1-ubyte',mnist_data_address{2});
save_mnist{3} = websave('t10k-images-idx3-ubyte',mnist_data_address{3});
save_mnist{4} = websave('t10k-labels-idx1-ubyte',mnist_data_address{4});

gunzip('*.gz')

% Adjust the datasets.
% Train dataset.
fprintf('Formattig and saving the MNIST datasets.\n')
train_file = {'train-images-idx3-ubyte','t10k-images-idx3-ubyte'};
mnist_input = [];
for file = 1:2
    fp = fopen(train_file{file}, 'rb');

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');

    numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
    numRows   = fread(fp, 1, 'int32', 0, 'ieee-be');
    numCols   = fread(fp, 1, 'int32', 0, 'ieee-be');

    images    = fread(fp, inf, 'unsigned char');
    images    = reshape(images, numCols * numRows, numImages);

    images    = double(images)' / 255;
    fclose(fp);
    mnist_input = [mnist_input;images];
end

% Test dataset.
test_file = {'train-labels-idx1-ubyte','t10k-labels-idx1-ubyte'};
mnist_target = [];
for file = 1:2
    fp = fopen(test_file{file}, 'rb');

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');

    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

    labels = fread(fp, inf, 'unsigned char');

    assert(size(labels,1) == numLabels, 'Mismatch in label count');

    fclose(fp);
    mnist_target = [mnist_target;labels];
end

% Save the datasets.
save('MNIST_input','mnist_input')
save('MNIST_target','mnist_target')
fprintf('MNIST data preprocessing is completed.\n')
clear

%% CIFAR-10
% Download and save the datasets. 
fprintf('Downloading CIFAR10 datasets... (184MB)\n')
cifar10_data_address = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
save_cifar10 = websave('cifar-10-data',cifar10_data_address);
untar(save_cifar10);

% Adjust the datasets.
fprintf('Formattig and saving the CIFAR10 datasets.\n')
cd cifar-10-batches-mat

cifar10_input =[];
cifar10_target =[];
for batch = 1:5
    load(['data_batch_' num2str(batch) '.mat'])
    cifar10_input = [cifar10_input;data];
    cifar10_target = [cifar10_target;labels];
end
load test_batch.mat
cifar10_input = [cifar10_input;data];
cifar10_target = [cifar10_target;labels];

cifar10_input = double(cifar10_input); 
cifar10_target = double(cifar10_target);

% save the datasets.
cd ../
save('CIFAR10_input','cifar10_input')
save('CIFAR10_target','cifar10_target')
fprintf('CIFAR10 data preprocessing is completed.\n')

% Remove files excepts datasets.
warning off
delete('*ubyte*','cifar-10*')
rmdir('cifar-10*','s')
cd ../
fprintf('Data processing finished successfully.\n')


