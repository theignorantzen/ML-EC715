function [net]=elm_standard(Xdata,Ydata,Opts,number_neurons,C,seed)

% elm_LB: this function allows to train a single hidden layer
% feedforward network  for regression with Moore-Penrose pseudoinverse of matrix.   
% Opts  : contains training options (look to example.m file) 
% Xdata : original not normalized Inputs(training plus testing)
% Ydata : original not normalized Targets(training plus testing)
% net   : contains ALL the iformation of the trained network
% C is for regularisation factor
%In the dats sets, Instances will be in the rows and attributes/variables will be in the columns
 
   
 %rand(seed,twister)   
rng(seed);
%%% save the important data caracteristics befor normalization

N1=min(min(Ydata));                 % save for denormalization
N2=max(max(Ydata));                 % save for denormalization

Xdata=scaledata(Xdata,0,1);
%Ydata=scaledata(Ydata,0,1);

 
%%% initialization
%number_neurons=Opts.number_neurons; % get number of neurons
            % get Application Type
                        % transform lables into binary codes
%%% Normalize your data according To ELM_Type

%%%% divide your data into training and testing sets according to training ratio
if Opts.fixed==1
    X=Xdata(1:60000,:);        % X  :    training inputs 
    Xts=Xdata(60001:end,:);    %Y  :    training targets
    Y=Ydata(1:60000,:);         %Xts:    testing inputs
    Yts=Ydata(60001:end,:);     %Yts:    testing targets
else
[X,Y,Xts,Yts]=divide_data(Opts.Tr_ratio,Xdata,Ydata);
%         X  :    training inputs
%         Y  :    training targets
%         Xts:    testing inputs
%         Yts:    testing targets
end


% save
net.C=C;
[Nsamples,Nfea]=size(X);
net.X=X;                % scaled training inputs
net.Y=Y;                % training targets
net.Xts=Xts;            % scaled testing inputs
net.Yts=Yts;            % testing target

%%%% encode lables for classification only

%%%% 1st step: generate a random input weights
input_weights=rand(number_neurons,size(X,2))*2-1;
tempH=input_weights*X';



%%%% 2nd step: calculate the hidden layer

switch lower(Opts.ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        
        case {'tansig'}
        %%%%%%%% tansigmoid basis function
        H = tansig(tempH);
        case {'logsig'}
        %%%%%%%% logsigmoid basis function
        H = logsig(tempH);
        case {'cos'}
        %%%%%%%% cosine basis function
        H = cos(tempH);
        case {'elliotsig'}
        %%%%%%%% elliot basis function
        H = elliotsig(tempH);
end




%%%% 3rd step: calculate the output weights beta
%B=pinv(H') * Y ; % Moore-Penrose pseudoinverse of matrix
%B=(eye(size(H,1))/C+H * H')\H*Y;  

if Opts.Regularisation==1
 if number_neurons<Nsamples
B=(eye(size(H,1))/C+H * H')\H*Y; 
%      
    else
   B=(((eye(size(H,2))/C+H' * H) \H')'*Y);   
  
 end
else
    B=pinv(H') * Y;% Moore-Penrose pseudoinverse of matrix
end
    
net.OW=B;        % save output weights
%%%% calculate the actual output of traning and testing 
Y_hat=(H' * B) ;%actual output/target of training data

tempH_test=input_weights*Xts';
 

switch lower(Opts.ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = double(hardlim(tempH_test));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);
        
        case {'tansig'}
        %%%%%%%% tansigmoid basis function
        H_test = tansig(tempH_test);
        case {'logsig'}
        %%%%%%%% logsigmoid basis function
        H_test = logsig(tempH_test);
        case {'cos'}
        %%%%%%%% cosine basis function
        H_test = cos(tempH_test);
        case {'elliotsig'}
        %%%%%%%% elliot basis function
        H_test = elliotsig(tempH_test);
end

Yts_hat=(H_test)'*B;


%%%% calculate the prefomance of training and testing

TrAccuracy=sqrt(mse(Y-Y_hat));       % RMSE for regression
TsAccuracy=sqrt(mse(Yts-Yts_hat));   % RMSE for regression
%Y_hat=scaledata(Y_hat,N1,N2);        % denormalization
%Yts_hat=scaledata(Yts_hat,N1,N2);    % denormalization
%Y=scaledata(Y,N1,N2);                % denormalization
%Yts=scaledata(Yts,N1,N2);            % denormalization
net.Y_hat=Y_hat;                     % estimated training targets
net.Yts_hat=Yts_hat;                 % estimated testing targets
net.Y=Y;
net.Yts=Yts;

    
% save data
net.min=N1;             % save the min value of Targets
net.max=N2;             % save the max value of Targets
net.Opts=Opts;          % save options
net.training_accuracy=TrAccuracy;  % training accuracy
net.testing_accuracy=TsAccuracy;  % testing accuracy
net.tempH=tempH;
net.Opts.ActivationFunction=Opts.ActivationFunction;

%net.label=label;

end

function dataout = scaledata(datain,minval,maxval)
% Program to scale the values of a matrix from a user specified minimum to a user specified maximum
dataout = datain - min(datain(:));
dataout = (dataout/range(dataout(:)))*(maxval-minval);
dataout = dataout + minval;
end
function[Inputs,Targets,TsInputs,TsTargets]=divide_data(trainRatio,x,y)
% 
% train_ratio=(number training samples/ number of samples in dataset)
% inputs   : training inputs
% targets  : training targets
% TsInputs : testing inputs
% TsTargets: testing targets

%%% calculate testing ratio
testRatio=1-trainRatio;
%%% get random dividing indexes 
[trainInd,valInd,testInd]=divideint(size(x,1),trainRatio,0,testRatio);
%%% divide  data
Inputs(1:length(trainInd),:)=x(trainInd,:); 
Targets(1:length(trainInd),:)=y(trainInd,:); 
TsInputs(1:length(testInd),:)=x(testInd,:); 
TsTargets(1:length(testInd),:)=y(testInd,:); 

end
