clear;clc
load Data_exam
Opts.fixed=0;  % 0 for random training
number_neurons=2000;   % Given in question
Opts.Tr_ratio=0.70;    % Given in question
%Opts.ActivationFunction='tribas';% Activation functions selection
Opts.Regularisation=1; % 1 for inverting with regularisation
%C=10^2;
seed=4509;% seed for random number
%% Training and testing
z=1;
activation={'sigmoid','sine','hardlim','tribas','radbas','tansig','logsig','cos','elliotsig'};
for x=1:9
    Opts.ActivationFunction=activation{x};
    C=10^2;
    for i = 1:5
    [net]= elm_standard(Xdata,Ydata,Opts,number_neurons,C,seed);
    Res(z).error = net.training_accuracy;
    Res(z).af = Opts.ActivationFunction; 
    Res(z).C = C;
    C = C*100;
    z = z+1;
    end
end


N = min([Res.error]);
for j = 1:z-1;
    if(Res(j).error==N)
        disp(Res(j))
        break;
    end
end


