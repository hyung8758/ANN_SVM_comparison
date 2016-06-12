% 2016, spring semester team project. 
% spreadTarget
% 
%                                                             Hyungwon Yang
%                                                                2016.06.10
%                                                                 EMCS labs


function new_target = spreadTarget(target)

train_target = 0.*ones(10, size(target, 1));
for n = 1: size(target, 1)
    train_target(target(n) + 1, n) = 1;
end;

new_target = train_target;