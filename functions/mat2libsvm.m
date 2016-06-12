% 2016, spring semester team project. 
% mat2libsvm
% 
%                                                             Hyungwon Yang
%                                                                2016.06.10
%                                                                 EMCS labs



function mat2libsvm(train_data,test_data,output_name)

% Data information.
data_num = size(train_data,1);
input_dim = size(train_data,2);

if nargin < 3
    output_name = 'mat2libsvm_output';
end

H = waitbar(0,'READY','CreateCancelBtn','setappdata(gcbf,''cancel'',1)'); 
% Writing text file.
txt=fopen([output_name '.txt'],'w','n','UTF-8');

for data = 1:data_num
    
    waitbar(data/data_num,H,['Writing ' output_name ' file...']);
    if getappdata(H,'cancel')
        fprintf('Converting process is interrupted.\n')
        delete([output_name '.txt'])
        break; 
    end
   
    fprintf(txt,'%d ',test_data(data));
    for dim = 1:input_dim
        if train_data(data,dim) ~= 0
            fprintf(txt,'%d:%g ', dim, train_data(data,dim));
        end
    end
    fprintf(txt,'\n');
end
fclose(txt);
delete(H)

