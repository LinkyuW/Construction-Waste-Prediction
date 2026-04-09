clc;
clear all;
[num] = xlsread(["广州.xlsx"]);
dataSize = size(num, 1); % 获取数据集的行数
cv = cvpartition(dataSize, 'HoldOut', 0.3); % 分配30%的数据作为测试集，剩余为训练集
idxTrain = training(cv); % 获取训练集的索引
idxTest = test(cv);     % 获取测试集的索引
x1 = num(:,1);
x2 = num(:,2);
x3 = num(:,3);
x4 = num(:,4);
x5 = num(:,5);
x6 = num(:,6);
x7 = num(:,7);
y = num(:,8);

%训练集
x1_t = x1(idxTrain);
x2_t = x2(idxTrain);
x3_t = x3(idxTrain);
x4_t = x4(idxTrain);
x5_t = x5(idxTrain);
x6_t = x6(idxTrain);
x7_t = x7(idxTrain);
y_t = y(idxTrain);
P = [x1_t,x2_t,x3_t,x4_t,x5_t,x6_t,x7_t,y_t];

%测试集
x1_te = x1(idxTest);
x2_te = x2(idxTest);
x3_te = x3(idxTest);
x4_te = x4(idxTest);
x5_te = x5(idxTest);
x6_te = x6(idxTest);
x7_te = x7(idxTest);
y_te = y(idxTest);
P2 = [x1_te,x2_te,x3_te,x4_te,x5_te,x6_te,x7_te,y_te];

%数据归一化
P_train = P(:,1:7)';
T_train = P(:,8)';
M = size(P_train,2);

P_test = P2(:,1:7)';
T_test = P2(:,8)';
N = size(P_test,2);

[P_train,ps_input] = mapminmax(P_train,0,1);
P_test = mapminmax("apply",P_test,ps_input);

[T_train,ps_output] = mapminmax(T_train,0,1);
T_test = mapminmax("apply",T_test,ps_input);
    

%数据平铺
P_train = double(reshape(P_train,7,1,1,M));
P_test = double(reshape(P_test,7,1,1,N));

T_train = T_train';
T_test = T_test';

%数据格式转换
for i = 1:M
    p_train{i,1} = P_train(:,:,1,i);
end
for i = 1:N
    p_test{i,1} = P_test(:,:,1,i);
end

%创建模型
numHiddenUnits =16;
LearnRateDropPeriod=800;  %乘法之间的纪元数由" LearnRateDropPeriod8控制
LearnRateDropFactor=0.5;  %乘法因子由参" LearnRateDropFactor"控制，
numFeatures =  7;   
numResponses =  1;  

layers = [ ...
    sequenceInputLayer(numFeatures)  
    lstmLayer(numHiddenUnits,'OutputMode','last') 
    reluLayer
    fullyConnectedLayer(numResponses) 
    regressionLayer];         
 
    options = trainingOptions('adam', ...%指定训练选项，求解器设置为adam， 1000轮训练。
        'MiniBatchSize',8,...
        'MaxEpochs',1000, ...     %最大训练周期为1000
        'GradientThreshold',1, ...   %梯度阈值设置为 1
        'InitialLearnRate',0.001, ...  %指定初始学习率 0.01
        'LearnRateSchedule','piecewise', ...  %每当经过一定数量的时期时，学习率就会乘以一个系数。
        'LearnRateDropPeriod', LearnRateDropPeriod, ...  
        'LearnRateDropFactor',LearnRateDropFactor, ...  %在50轮训练后通过乘以因子 0.5 来降低学习率。
        'Verbose',0, ...   %如果将其设置为true，则有关训练进度的信息将被打印到命令窗口中,0即是不打印 。
        'Plots','training-progress');   %构建曲线图 ，不想构造就将'training-progress'替换为none
 
 
net = trainNetwork(p_train,T_train,layers,options);    %训练神经网络
%save('LSTM_net', 'net');            %将net保存为LSTM_net


%仿真预测
t_sim1 = predict(net,p_train);
t_sim2 = predict(net,p_test);


%数据反归一化
T_sim1 = mapminmax('reverse',t_sim1,ps_output);
T_sim2 = mapminmax('reverse',t_sim2,ps_output);
X_t = 1:38;
X_te= 39:54;
hold on;
plot(X_t,T_sim1,'r');
plot(X_t,y_t,'b');
plot(X_te,T_sim2,'r');
plot(X_te,y_te,'b');
legend("预测值","真实值");

%计算训练集评价指标
mae_train = mean(abs(y_t - T_sim1));
mse_train = mean((y_t - T_sim1).^2);
rmse_train = sqrt(mse_train);
sstot_train = sum((y_t - mean(y_t)).^2); % 总平方和
ssres_train = sum((y_t - T_sim1).^2);      % 残差平方和
r2_train = 1 - ssres_train / sstot_train;                 % R^2

% 计算验证集的评价指标
mae_val = mean(abs(y_te - T_sim2));
mse_val = mean((y_te - T_sim2).^2);
rmse_val = sqrt(mse_val);
sstot_val = sum((y_te - mean(y_te)).^2);     % 总平方和
ssres_val = sum((y_te - T_sim2).^2);          % 残差平方和
r2_val = 1 - ssres_val / sstot_val;                      % R^2

% 打印结果
fprintf('训练集数据的MAE为：%f\nRMSE为：%f\nMSE为：%f\nR^2为：%f\n', mae_train, rmse_train, mse_train, r2_train);
fprintf('验证集数据的MAE为：%f\nRMSE为：%f\nMSE为：%f\nR^2为：%f\n', mae_val, rmse_val, mse_val, r2_val);