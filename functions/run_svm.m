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
    
    try
        % Test whether svmtrain is working correctly.
        svmtrain
        
    catch
        
        % Data selection.
        warning(['svmtrain is not working properly on the matlab. '...
            'It will be running on the ms-dos instead.'])
        
        
        cd data
        train_name = [name '_train.txt'];
        test_name = [name '_test.txt'];
        model_name = [name '.model'];
        result_name = [name '_result.txt'];
        movefile(train_name,'../windows_svm')
        movefile(test_name,'../windows_svm')
        
        % Check whether text format datasets exist.
        train_check = dir(train_name);
        test_check = dir(test_name);
        % For train.
        if isempty(train_check)
            
            fprintf('%s dataset is missing.\n Transoforming the mat file to text file.',train_name)
            % Transform the datasets.
            train_number = input('Please set the trainset numbers: (MNIST: 60000 / CIFAR10: 50000) '); 
            test_number = input('Please set the testset numbers: (MNIST:10000 / CIFAR10: 10000) ');
            mat2libsvm(train_input(1:train_number,:),test_target(1:train_number,:),train_name)
        end
        % For test.
        if isempty(test_check)
            
            fprintf('%s dataset is missing.\n Transoforming the mat file to text file.',test_name)
            % Transform the datasets.
            train_number = input('Please set the trainset numbers: (MNIST: 60000 / CIFAR10: 50000) '); 
            test_number = input('Please set the testset numbers: (MNIST:10000 / CIFAR10: 10000) ');
            mat2libsvm(train_input(train_number+1:train_number+test_number,:),test_target(train_number+1:train_number+test_number,:),test_name)
        end
        frpintf('text form datasets are ready.\n')
        
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

    end
    
    % Train data.
    fprintf('Training model...\n')
    model = svmtrain(train_target,train_input,opt);
    
    % Predict data.
    [~, accuracy, ~] = svmpredict(test_target, test_input, model);
    fprintf('svm training is finished..\n')
    
    
else
    error('Linux is not supported yet.')
end