clear;clc;

%% Load Data
load Dataset
Xdata=Antenna_microstrip(:,1:4); %input data
Ydata=Antenna_microstrip(:,5);   %output/target data

%% Define Options
number_neurons = 200;
Opts.fixed=0; %random testing/training data
Opts.Tr_ratio=0.80; 
Opts.ActivationFunction='tribas';
Opts.Regularisation=1;  %1 for Ridge regression
C = 10^13;                 %reqd for regularization
seed = 5200;

%% Function call for training and testing
Res = zeros(10,2);
for i = 1:10
    seed = seed + 1;
    [net]= ELM_func(Xdata,Ydata,Opts,number_neurons,C,seed);
    Res(i,1) = net.testing_accuracy;
    Res(i,2) = seed;  
end

N = min(Res(:,1));
for j = 1:10
    if(Res(j,1)==N)
        disp(Res(j,2));
        disp(Res(j,1));
    end
end
