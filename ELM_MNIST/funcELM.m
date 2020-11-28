function [net] = funcELM(trainX,trainY,testX,testY,Opts,number_neurons,C,seed)
rng(seed);
trainX=scaledata(trainX,0,1);
testX=scaledata(testX,0,1);
[Nsamples,Nfea]=size(trainX);
net.trainX=trainX;
net.testX=testX;
net.trainY=trainY;
net.testY=testY;

%1st step: gen random wts
input_weights=rand(number_neurons,size(trainX,2))*2-1;
tempH=input_weights*trainX';

%2nd step: calculate hidden layer
switch lower(Opts.ActivationFunction)
    case {'tribas'}
        %%%trangular bias 
        H = tribas(tempH);
end

%3rd step: calculate output weights
if number_neurons<Nsamples
    B=(eye(size(H,1))/C+H * H')\H*trainY;      
else
    B=(((eye(size(H,2))/C+H' * H) \H')'*trainY);   
end
 
net.OW=B;

%4th step: calculate the actual output of training and testing data
trainY_hat=(H' * B) ;%actual output/target of training data

tempH_test=input_weights*testX';
switch lower(Opts.ActivationFunction)
    case {'tribas'}
        %%%trangular bias 
        H_test = tribas(tempH_test);
end
testY_hat = (H_test)'*B;

% calculating the performance of training and testing
TrRMSE=sqrt(mse(trainY-trainY_hat));       
TsRMSE=sqrt(mse(testY-testY_hat));

net.TrRMSE=TrRMSE;
net.TsRMSE=TsRMSE;
end

function dataout = scaledata(datain,minval,maxval)
% Program to scale the values of a matrix from a user specified minimum to a user specified maximum
dataout = datain - min(datain(:));
dataout = (dataout/range(dataout(:)))*(maxval-minval);
dataout = dataout + minval;
end