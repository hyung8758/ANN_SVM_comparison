% 2016, spring semester team project. 
% run_svm
% 
%                                                             Hyungwon Yang
%                                                                2016.06.10
%                                                                 EMCS labs


function accuracy = run_svm(param)

% Retrieve the parameters.
name = param.name;
train_input = param.train_input;
train_target = param.train_target;
test_input = param.test_input;
test_target = param.test_target;
opt = param.option;

if ismac == 1
    
    % Train data.
    fprintf('Training model...\n')
    model = svmtrain(train_target,train_input,opt);
    
    % Predict data.
    [~, accuracy, ~] = svmpredict(test_target, test_input, model);
    fprintf('svm training is finished..\n')
    
    
elseif ispc == 1
    
    % Data selection.
    cd data
    train_name = [name '_train.txt'];
    test_name = [name '_test.txt'];
    model_name = [name '.model'];
    result_name = [name '_result.txt'];
    movefile(train_name,'../windows_svm')
    movefile(test_name,'../windows_svm')
    
    % Train data.
    cd ../windows_svm
    fprintf('Training model...\n')
    system(['svm-train ' opt ' ' train_name ' ' model_name])
    
    % Predict data.
    fprintf('Predicting data with generated model...\n')
    system(['svm-predict ' test_name ' ' model_name ' ' 'tmp.txt' ' > ' result_name])
    
    fid = fopen(result_name,'r');
    data = textscan(fid,'%s','delimiter',' ');
    fclose(fid);
    accuracy = data{1}{3};
    
    % Move result file.
    movefile(train_name,'../data/')
    movefile(test_name,'../data/') 
    movefile(model_name,'../data/') 
    movefile(result_name,'../data/')
    delete('tmp.txt')
    cd ../
    fprintf('svm model and result files are saved in the data folder.\n')
    
else
    error('Linux is not supported yet.')
end