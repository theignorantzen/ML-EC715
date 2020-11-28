function [net]=ELM_func(Xdata,Ydata,Opts,number_neurons,C,seed)

rng(seed); %Control random number generation
N1 = min(min(Ydata));  %save for denormalization
N2 = max(max(Ydata));  %save for denormalization

Xdata = scaledata(Xdata,0,1);

% Dividing the data into traing and testing sets
if Opts.fixed==1
    X=Xdata(1:60000,:);        % X : training inputs 
    Xts=Xdata(60001:end,:);    %Xts: testing inputs
    Y=Ydata(1:60000,:);        %Y  : training targets
    Yts=Ydata(60001:end,:);    %Yts: testing targets
else
    [X,Y,Xts,Yts]=divide_data(Opts.Tr_ratio,Xdata,Ydata);
end

% save
[Nsamples,Nfea]=size(X);
%net.X=X;                % scaled training inputs
%net.Y=Y;                % training targets

% 1st step: Generating random input weights
input_weights=rand(number_neurons,size(X,2))*2-1;
tempH=input_weights*X';

% 2nd step: Calculate the hidden layer
switch lower(Opts.ActivationFunction)
    case {'sig','sigmoid'}
        H = 1 ./ (1 + exp(-tempH));
    case {'tribas'}
        H = tribas(tempH);
end

% 3rd step: Calculate output weights
if Opts.Regularisation==1     % Ridge regression method
 if number_neurons<Nsamples
    B=(eye(size(H,1))/C+H * H')\H*Y;       
 else
    B=(((eye(size(H,2))/C+H' * H) \H')'*Y);   
 end
else
    B=pinv(H') * Y;% Moore-Penrose pseudoinverse of matrix
end

net.OW=B;  %save o/p weights
Y_hat = (H'*B); % Obtained output of training data using the network with B weights



% Training Complete. Testing starts here.
tempH_test=input_weights*Xts';

switch lower(Opts.ActivationFunction)
    case {'sig','sigmoid'}
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'tribas'}
        H_test = tribas(tempH_test);
end

Yts_hat=(H_test)'*B;

% Calculating the performance of the training and testing and saving values
TrAccuracy=sqrt(mse(Y-Y_hat));       % RMSE for regression
TsAccuracy=sqrt(mse(Yts-Yts_hat));   % RMSE for regression


%save data
%net.Y_hat=Y_hat;                     % estimated training targets
%net.Yts_hat=Yts_hat;                 % estimated testing targets
%net.Y=Y;
%net.Yts=Yts;
%net.min=N1;             % save the min value of Targets
%net.max=N2;             % save the max value of Targets
%net.Opts=Opts;          % save options
net.training_accuracy=TrAccuracy;  % training accuracy
net.testing_accuracy=TsAccuracy;  % testing accuracy
%net.tempH=tempH;
%net.Opts.ActivationFunction=Opts.ActivationFunction;

end

% Function to scale the values of a matrix
function dataout = scaledata(datain,minval,maxval)
dataout = datain - min(datain(:));
dataout = (dataout/range(dataout(:)))*(maxval-minval);
dataout = dataout + minval;
end

function[Inputs,Targets,TsInputs,TsTargets]=divide_data(trainRatio,x,y)
testRatio=1-trainRatio;
%%% get random dividing indexes 
[trainInd,valInd,testInd]=divideint(size(x,1),trainRatio,0,testRatio);
%%% divide  data
Inputs(1:length(trainInd),:)=x(trainInd,:); 
Targets(1:length(trainInd),:)=y(trainInd,:); 
TsInputs(1:length(testInd),:)=x(testInd,:); 
TsTargets(1:length(testInd),:)=y(testInd,:); 
end


