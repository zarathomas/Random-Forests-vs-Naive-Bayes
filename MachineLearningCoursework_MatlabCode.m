%% Random Forest Matlab Code 
%% Load Imported Data into Matlab 
load('df.mat')
df.Properties.VariableNames{19} = 'group';

%% Isolate Payment Data and Default Next Payment(Group)
% Remove ID column from dataset
x = df(:,2:19)
% m and n specify the number of rows and colums in the dataset
[m,n] = size(x)

%% Split the data into training and test sets (holdout method)
% p represents percentage of data held for Test Set(30%)
p=0.3;
cvpt = cvpartition(m, 'Holdout', p);
dataTrain = x(training(cvpt),:);
dataTest = x(test(cvpt),:);
%% Split the data into training and test sets (holdout/stratified method)
%Partition data into 70% for Training Data and 30% for Test Data 
p=0.3;
% Specify target column in data 
group = x.group;
cvpt = cvpartition(group, 'Holdout', p);
%Store training set and test set as dataTrain and dataTest respectively
dataTrain = x(training(cvpt),:);
dataTest = x(test(cvpt),:);

%% Use Random Forests Classifier
% nTrees is where the number of trees is specified.
nTrees = 200;
for i = nTrees
    tic;
    B = TreeBagger(i,dataTrain, 'group', 'Method', 'classification','OOBPredictorImportance','on');
    err = error(B,dataTest);
    avgerr = mean(err);
    toc;   
    elapsedTime = toc; 
end

%Print average mean error for 1:100 trees 
avgerr  
elapsedTime


%% Predict the Output 
TrueTest = (dataTest{:,18});
TrueData = (dataTrain{:,18});
%Training Data: 
%predictedGroups = str2double(predict(B,dataTrain));
%Test Data: 
predictedGroups = str2double(predict(B,dataTest));

%Cost matrix associated with misclassification 
costMat = B.Cost;

%% Predict Output and Scores 
[Yfit, scores] = predict(B,dataTrain)
Yfit = str2double(Yfit)
%% Visualise The Results in Table
Comparison = table(TrueData,predictedGroups,'VariableNames', {'ObservedValue', 'PredictedValue'});

%% Evaluate importance of features 
nTrees = 10:20; 
for i = nTrees
    B = TreeBagger(i,dataTrain, 'group', 'Method', 'classification','OOBPredictorImportance','on', 'nvartosample','all','PredictorSelection','curvature');
    imp = B.OOBPermutedPredictorDeltaError;
end
[~,idximp]= sort(imp);

%% Evaluate importance of features (loop)
nTrees = 1:100;
C = [];
for i = nTrees
        B = TreeBagger(i,dataTrain, 'group', 'Method', 'classification','OOBPredictorImportance','on', 'nvartosample','all','PredictorSelection','curvature');
        imp = B.OOBPermutedPredictorDeltaError;
        C = [C;imp];
end

%% Try K Fold Cross Valdiation with 10 folds 
Y = table2array(x(:,18))
X = table2array(x(:,1:17))
%cp = cvpartition(Y,'k',10); % Stratified cross-validation

%% Cross Validation for Random Forests
%CVO = cvpartition(Y,'k',10);
%data partition
tic
    cp = cvpartition(Y,'k',10); %10-folds
    nTrees = 200;
    for i = nTrees
        tic;
        %Model 
        %prediction function
        classF = @(XTRAIN,ytrain,XTEST)(predict(TreeBagger(i,XTRAIN,ytrain,'Method', 'classification','OOBPredictorImportance','on'),XTEST));
        %missclassification error 
        missclasfError(i) = crossval('mcr',X,Y,'predfun',classF,'partition',cp);
        toc;
        elapsedTime(i) = toc; 
     end
    
    %Print Misclassification Error
    missclasfError 
    elapsedTime;
    
%% Predict the Output 
TrueTest = (dataTest{:,18});
TrueData = (dataTrain{:,18});
%Training Data: 
predictedGroups = str2double(predict(B,dataTrain));
%Test Data: 
predictedGroups = str2double(predict(B,dataTest));

%Cost matrix associated with misclassification 
costMat = B.Cost;


%% Try K Fold Cross Valdiation with 10 folds
%Split data into target variables(Y) and predictor variables(X)
Y = table2array(x(:,18))
X = table2array(x(:,1:17))
%cp = cvpartition(Y,'k',10); % Stratified cross-validation

%% Cross Validation for Random Forests
%Partition data for cross validation using the cvpartition function
%Use tic-toc to time model run 
tic
    cp = cvpartition(Y,'k',10); %10-folds
    nTrees = 1;
    for i = nTrees
        tic;
        %Model 
        %prediction function
        classF = @(XTRAIN,ytrain,XTEST)(predict(TreeBagger(i,XTRAIN,ytrain,'Method', 'classification','OOBPredictorImportance','on'),XTEST));
        %missclassification error 
        missclasfError(i) = crossval('mcr',X,Y,'predfun',classF,'partition',cp);
        toc;
        elapsedTime(i) = toc; 
     end
    
    %Print Misclassification Error
    missclasfError 
    %Print time taken to run model 
    elapsedTime;

%% Naive Bayes Matlab code
clear all; 

%%Step 1
%%Import uci dataset - for documentation, refer: https://uk.mathworks.com/help/matlab/ref/importoptions.setvartype.html

filename  = 'uci_default.csv';
%Identify variable names 
opts = detectImportOptions(filename);
%Correct variable data types 
opts = setvartype(opts,{'SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'},'categorical');
%Read csv file as a table structure, with correct variable names and data types
M = readtable(filename, opts);
%Drop 'ID' column (Column 1), which has no predictive value. 
M = M(:,[2:25]);

%%Step 2
%%Fit a preliminary naive bayes model
%Holdout method of cross-validation (70%-30% data split for training and
%testing). We manually implement this to  access to in-sample and out-of-sample training errors. 
train = M(1:21000,:);
test = M(21001:30000,:);
%Assign predictor data columns
X_M = train(:, 1: 23);
%Assign response data column
Y_M = train(:,24);

%%Note: We use Matlab's CrossVal method to implement K-Folds cross-validation. This only allows reporting of average
%%out-of-sample classification error rates. X and Y data do not need to be manually paritioned. 
%Assign predictor data columns to 'X'  (for crossval methods)
X = M(:, 1:23); 
%Assign response data column to 'Y' (for crossval methods)
Y = M(:,24);

%Fit the naive bayes model. 
%i.Holdout model.
man_mdl = fitcnb(X_M, Y_M, 'ClassNames', {'1','0'});
isGenRate = resubLoss(man_mdl,'LossFun', 'classiferror');%In-sample error rate (Ans: 0.3143)
oosGenRate = loss(man_mdl, test(:,1:23), test(:,24)); %Out-of-sample error rate (Ans:0.2861)

%ii.K-Folds model 
orig_mdl = fitcnb(X, Y, 'ClassNames', {'1','0'}, 'Crossval', {'On'});
orig_error = kfoldLoss(orig_mdl); %Out-of-sample k-fold classification error (Ans: 0.3413) 


%%Step 3
%%Hyperparameter tuning 
%%Orig_model assumes a 'normal' distribution for continuous predictors, but
%%these can  be modeled using a kernel smoothing density estimate. There are 4 types of
%%kernels: Normal, Epanechnikov, Box, Triangle

%Fit a naive bayes model with all continuous features modeled as 'Kernel: Normal' distributions
distribution = {'kernel','mvmn','mvmn','mvmn','kernel','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel'}

%i. Holdout model
kern_norm_man_mdl = fitcnb(X_M, Y_M, 'Distribution', distribution, 'ClassNames', {'1','0'});
isGenRate_kern_n = resubLoss(kern_norm_man_mdl,'LossFun', 'classiferror'); %In-sample error rate (Ans: 0.2119)
oosGenRate_kern_n = loss(kern_norm_man_mdl, test(:,1:23), test(:,24)); %Out-of-sample error rate (Ans: 0.2084)

%ii. Crossvalidated model 
kern_norm_mdl = fitcnb(X, Y, 'Distribution', distribution, 'ClassNames', {'1','0'}, 'Crossval', {'On'});
kern_norm_error = kfoldLoss(kern_norm_mdl); % (Ans: 0.2115) 

%%Step 3.A.
%%A manually coded Exhaustive ('Brute Force') Grid Search for Kernel parameter selection
%Declare the four types of kernel distributions.
k_all_val = {'normal' 'epanechnikov' 'box' 'triangle'};

%%Note: there are 14 features with continuous data, but 6 are pay amounts and 6 are bill amounts.
%%For modeling purposes, we model the pay amount and the bill amount features with the same respective distribution. 
%%This leaves 4 types of continuous features: Limit, Age, Pay Amount (6 features), Bill Amount (6 features) 

%Create a matrix of all combinations of the 4 kernel distributions, taken 4 at a time
perms_k = nchoosek(repmat(k_all_val, 1,4),4);
%Remove duplicates. Convert to table structure in order to apply 'unique' function 
perms_k = cell2table(perms_k);
perms_k = unique(perms_k);
%Revert to matrix structure
perms_k = table2array(perms_k); 
%Calculate size of matrix 
size_k = size(perms_k); 

%Fit the naive bayes model for every possible combination of Kernel distributions. Record classification error. 
%i. Holdout model
%Declare an empty container
kfold_iis_map = containers.Map()
kfold_oos_map = containers.Map()

for value = 1:size_k(1)
    k_val = perms_k(value, :);
    k_dist = {k_val{1}, 'NULL', 'NULL', 'NULL', k_val{2}, 'NULL','NULL','NULL','NULL','NULL','NULL', k_val{3},k_val{3},k_val{3},k_val{3},k_val{3},k_val{3},k_val{4},k_val{4},k_val{4},k_val{4},k_val{4},k_val{4} }
    k_man_mdl = fitcnb(X_M, Y_M, 'Distribution', distribution, 'Kernel', k_dist,'ClassNames', {'1','0'});
    iisGenRate_K = resubLoss(k_man_mdl,'LossFun', 'classiferror');
    oosGenRate_K = loss(k_man_mdl, test(:,1:23), test(:,24));
    kfold_iis_map(strjoin(k_val)) = iisGenRate_K; 
    kfold_oos_map(strjoin(k_val)) = oosGenRate_K; 
end

%View output. Findings: 
keys(kfold_iis_map);
values(kfold_iis_map);
keys(kfold_oos_map);
values(kfold_oos_map);

%ii. Crossvalidated model.
kfold_map = containers.Map()
for value = 1:size_k(1)
    k_val = perms_k(value, :);
    k_dist = {k_val{1}, 'NULL', 'NULL', 'NULL', k_val{2}, 'NULL','NULL','NULL','NULL','NULL','NULL', k_val{3},k_val{3},k_val{3},k_val{3},k_val{3},k_val{3},k_val{4},k_val{4},k_val{4},k_val{4},k_val{4},k_val{4} }
    mdl = fitcnb(X, Y, 'Distribution', distribution, 'Kernel', k_dist,'ClassNames', {'1','0'}, 'Crossval', {'On'});
    class_loss = kfoldLoss(mdl);
    kfold_map(strjoin(k_val)) = class_loss;     
end

%View output. Finding: lowest classification error: 0.2037 ('box box normal box')
keys(kfold_map), values(kfold_map);

%Save output as csv file 
keys = kfold_map.keys;
keys = keys(:); 
values = kfold_map.values;
values = values(:); 
kfolds_tab = horzcat(keys, values);  
kfolds_tab = cell2table(kfolds_tab, 'VariableNames', {'Keys', 'Values'});
writetable(kfolds_tab,'Kernel_Permutations_Error_Results');

%%Step 3.B. Bayesian Optimisation for Kernel parameter selection 
Kernel1 = optimizableVariable('Kernel1',{'normal', 'box', 'epanechnikov', 'triangle'},'Type','categorical');
Kernel2 = optimizableVariable('Kernel2',{'normal', 'box', 'epanechnikov', 'triangle'},'Type','categorical');
Kernel3 = optimizableVariable('Kernel3',{'normal', 'box', 'epanechnikov', 'triangle'},'Type','categorical');
Kernel4 = optimizableVariable('Kernel4',{'normal', 'box', 'epanechnikov', 'triangle'},'Type','categorical');
distribution2 = {'kernel1','mvmn','mvmn','mvmn','kernel2','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','kernel3','kernel3','kernel3','kernel3','kernel3','kernel3','kernel4','kernel4','kernel4','kernel4','kernel4','kernel4'}

%i. Holdout model
bayes_mdl = fitcnb(X_M,Y_M,'Distribution', distribution2,'ClassNames', {'1','0'})
fun_iis= @(x)resubLoss(bayes_mdl,'LossFun', 'classiferror');
fun_oos = @(x)loss(bayes_mdl, test(:,1:23), test(:,24));
results_iis = bayesopt(fun_iis,[Kernel1, Kernel2, Kernel3, Kernel4], 'Verbose',1); 
results_oos = bayesopt(fun_oos,[Kernel1, Kernel2, Kernel3, Kernel4], 'Verbose',1); 

%ii. Crossvalidated model
fun = @(x)kfoldLoss(fitcnb(X,Y,'Distribution', distribution2,'ClassNames', {'1','0'}, 'Crossval', {'On'}));
results = bayesopt(fun,[Kernel1, Kernel2, Kernel3, Kernel4], 'Verbose',1); %Total elapsed time: 3270.8359 seconds; 30 functions evaluated;Best observed function value: 0.21147; (epanechnikov,epanechnikov, normal,epanechnikov)




