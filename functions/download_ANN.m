% 2016, spring semester team project. 
% download_ANN
% 
%                                                             Hyungwon Yang
%                                                                2016.06.10
%                                                                 EMCS labs


function download_ANN()

% download ANN algorithm
filename = 'matlab-mnist-two-layer-perceptron';
new_filename = 'matlab_ann';

if ispc == 1
    websave(filename','https://github.com/davidstutz/matlab-mnist-two-layer-perceptron/archive/master.zip');
    unzip([filename '.zip'])
    movefile([filename '-master'],filename)
    delete([filename '.zip'])
else
    system('git clone https://github.com/davidstutz/matlab-mnist-two-layer-perceptron');
end

cd(filename)
delete('loadMNIST*','t10k-*','train-*')
cd ../
movefile(filename,new_filename)
addpath(new_filename)