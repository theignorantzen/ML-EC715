clear; clc;
load mnist
Opts.ActivationFunction = 'tribas';
number_neurons=1000;
C = 10^15;
seed=5000;
trainX = double(trainX)/255;
trainY = double(trainY)/255;
testX = double(testX)/255;
testY = double(testY)/255;
[net] = funcELM(trainX, trainY', testX, testY', Opts, number_neurons, C, seed);
net