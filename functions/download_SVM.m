% 2016, spring semester team project. 
% download_SVM
% 
%                                                             Hyungwon Yang
%                                                              2016. 06. 10
%                                                                 EMCS labs


function download_SVM()

% Download SVM algorithm
filename = 'libsvm';
new_filename = '_svm';

if ispc == 1
    websave(filename,'https://github.com/cjlin1/libsvm/archive/master.zip');
    unzip([filename '.zip'])
    movefile([filename '-master'],filename)
    delete([filename '.zip'])
elseif ismac == 1
    system('git clone https://github.com/cjlin1/libsvm.git');
else 
    error('Linux is not supported yet.')
end

% Install svm.
if ispc == 1
    new_filename = 'windows_svm';
    cd(filename)
    movefile('windows',['../' new_filename])
    
else
    new_filename = 'matlab_svm';
    cd(fullfile(filename,'matlab'))
    try
        make
    catch 
        fprintf(['Error occurred during compiling codes,\nPlease find the problem '...
        'manually and try again.\n'])
    end
    cd ../
    movefile('matlab',['../' new_filename])
    
end

% Remove other redundent sources.
cd ../
rmdir(filename,'s')
addpath(new_filename)


