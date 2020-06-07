clear;
clc

%% generate data
% normal data
rng(1);
eta = randn(1,500);
ee = randn(1,500);

for i = 1 : 500
    temp = [1.5 1.8 1.3]'*eta(i) +[0.28 0.38 0.42]'*ee(i);
    data(:,i) = temp';
end

U = [1.57 2.37 1.8;2.73 1.05 1.4;1.22 1.60 2.4]*data+[0.35 0.42 0.315]'*ee;

X = [data;U];


% testing data
eta = randn(1,500);
ee =  randn(1,500);
for i = 1 : 500
    temp = [1.5 1.8 1.3]'*eta(i) +[0.28 0.38 0.42]'*ee(i);
    data(:,i) = temp';
end

U = [1.57 2.37 1.8;2.73 1.05 1.4;1.22 1.60 2.4]*data+[0.35 0.42 0.315]'*ee;
XT = [data;U];

% add step changes
for i = 201 : 500
    XT(3,i) = XT(3,i) + 2.5;
    XT(4,i) = XT(4,i) + 1.22;
end

%% Perform PCA 

[EigVec,~,Lat] = pca(X');
XT_d = EigVec(:,1:2)'*XT;
figure;
plot(XT_d(1,1:200),XT_d(2,1:200),'o');title('PCA')
hold on
plot(XT_d(1,201:end),XT_d(2,201:end),'ro');

B = EigVec(:,1:2);




%% DNOM
%Init network with one hidden layer of 7 nodes
net = fitnet(7);
% maximum number of iterations
MaxIter = 10;
for i = 1 : MaxIter
    % training network
    [net,tr] = train(net,X,B'*X);
    % perform svd 
    G = sim(net,X);
    [UU,ss,VV] = svd(X*G');
    BB = UU(:,1:2)*VV';
    if norm(BB-B)<0.001
        i
        break;
    else
        B = BB;
    end
end

XT_dnom = sim(net,XT);
figure;
plot(XT_dnom(1,1:200),XT_dnom(2,1:200),'o');title('DNOM');
hold on
plot(XT_dnom(1,201:end),XT_dnom(2,201:end),'ro');


%% KPCA with Gaussian Kernel and Kernel parameter is 4


options.KernelType = 'Gaussian';
options.t = 4;
options.ReducedDim = 7;
[eigvector,eigvalue] = KPCA(X',options);
Ktest = constructKernel(XT',X',options);
Y = Ktest*eigvector;
Y = Y';
figure
plot(Y(1,1:200),Y(2,1:200),'o');title('KPCA');
hold on
plot(Y(1,201:end),Y(2,201:end),'ro');