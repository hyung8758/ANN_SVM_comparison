% 2016, spring semester team project. 
% ANNtrain
% 
%                                                             Hyungwon Yang
%                                                                2016.06.10
%                                                                 EMCS labs



function [net,accuracy] = ANNtrain(net,input,target)

% Train the datasets.
[net,tr] = train(net,input,target);

outputs = net(input);
errors = gsubtract(target,outputs);
performance = perform(net,target,outputs);


% test set
tInd = tr.testInd;
testOutputs = net(input(:,tInd));
testPerform = perform(net,target(:,tInd),testOutputs);

% Calculate accuracy.
A = zeros(length(testOutputs),1);
B = zeros(length(testOutputs),1);
count = 0;
testTarget = target(:,tInd);
for i = 1:length(testOutputs)
    A(i,1) = (find(testOutputs(:,i) == max(testOutputs(:,i))));
    B(i,1) = (find(testTarget(:,i) == 1));
    if A(i,1) == B(i,1)
        count = count + 1;
    end
end

accuracy = count / length(testOutputs) * 100;

