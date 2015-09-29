%% Set options and stuff
clear all
addpath(genpath('.'))
rmpath(genpath('/usr/local/osl1.4.0/'))
rmpath(genpath('/net/aton/data/OHBA/fieldtrip-20141020/'))
addpath(genpath('../HMM-MAR-scripts/'))
ndim = 10; N = 1; Ttrial = 3000;
TfMRI = Ttrial * ones(N,1);
K = 3; p = 3;
HzSignal = 1; HzfMRI = 1;
covtype = 'full';
zeroBmean = 0;
constantHRF = 1;
StatePermanency = 50;
noiseFactor = [0.01 0.25];

%% simulate model and data

% model
[train.H,~,train.meanB,train.covB] = HRFbasis(p,HzSignal,zeroBmean);
L = size(train.H,2);
Tsignal = round((TfMRI-1) * (HzSignal/HzfMRI)) + L;
hmm_orig=simmodel(TfMRI,Tsignal,p,ndim,K,HzSignal,covtype,zeroBmean,constantHRF,StatePermanency,noiseFactor,train); clear train
% data 
data.T = TfMRI;
data.Hz = HzfMRI;

for tr=1:N
   hmm_orig.HRF(tr).sigma.rate(:) = 10e-15; 
end

[data.Y,X_orig,~,Gamma_orig]=simdata(hmm_orig,[],TfMRI,HzfMRI,HzSignal,0);
save('/tmp/X_orig.mat','X_orig')
save('/tmp/hmm_orig.mat','hmm_orig')
save('/tmp/Gamma_orig.mat','Gamma_orig')
save('/tmp/data.mat','data')

%% some plots
figure
subplot(2,2,1);plot(X_orig(1:1000,1));
subplot(2,2,2);plot(data.Y(:,1));
for i=1:N, A(i,:) = hmm_orig.HRF(i).sigma.rate ./ hmm_orig.HRF(i).sigma.shape; end
subplot(2,2,3);imagesc(A)
for i=1:N, A(i,:) = hmm_orig.HRF(i).alpha.rate ./ hmm_orig.HRF(i).alpha.shape; end
subplot(2,2,4);imagesc(A)


%% Init Gamma

H = HRFbasis(p,HzSignal,zeroBmean); L = size(H,2);
T = round((TfMRI-1) * (HzSignal/HzfMRI)) + L;
options.Gamma = zeros(sum(T),K);
for tr=1:length(T)
    t0 = sum(T(1:tr-1)); t1 = sum(T(1:tr));
    GammaInit = rand(T(tr),K);
    options.Gamma(t0+1:t1,:) = GammaInit ./ repmat(sum(GammaInit,2),1,K);
end
save('/tmp/GammaInit.mat','GammaInit')
clear H t0 t1 GammaInit

%% run model 

options.K = K;
options.p = 3; 
options.cyc = 1000;
options.covtype = 'full';
options.zeroBmean = zeroBmean;
options.Hz = HzSignal;

[hmm, Gamma, ~, ~, X, fehist] = hmmfmri (data,options);

%% plot X and Gamma

figure(1);
ranget = 101:600; 
variables = [1 3 5 9];
for v=1:4
subplot(2,2,v); plot(ranget,[X_orig(ranget,variables(v)) X.mu(ranget,variables(v)) ] ) ; 
xlim([ranget(1) ranget(end)]); legend('Truth','posterior')
end
figure(2)
subplot(2,1,1); plot(ranget,Gamma(ranget,:)); ylim([-0.1 1.1])
subplot(2,1,2); plot(ranget,Gamma_orig(ranget,:)); ylim([-0.1 1.1])
[ diag(corr(X_orig,X.mu)) ]

%% plotting HRF
figure(1)
for i=1:N
    subplot_tight(N,3,(i-1)*3+1,[0.05 0.05]) 
    imagesc(hmm_orig.HRF(i).B.mu); colorbar
    subplot_tight(N,3,(i-1)*3+2,[0.05 0.05]) 
    imagesc(hmm.HRF(i).B.mu); colorbar
    subplot_tight(N,3,(i-1)*3+3,[0.05 0.05]) 
    imagesc(abs(hmm.HRF(i).B.mu - hmm_orig.HRF(i).B.mu)); colorbar
end


%% testing obsinit to see if it deconvolves well enough 

% simulating latent signal 
X2 = randn(sum(T),ndim);
for n=1:ndim
    X2(:,n) = smooth(X2(:,n),10);
end

% computing fmri signal
Y2 = zeros(sum(TfMRI),ndim);
hmm0 = {}; hmm0.train = {};
[hmm0.train.H,hmm0.train.meanH] = HRFbasis(3,data.Hz,1);
[hmm0.train.I1,hmm0.train.I2] = initindexes(data.T,length(meanH),data.Hz,options.Hz);
hmm0.train.covtype = 'full';
hmm0.train.init_iterations = 50;

for tr=1:N
    t0 = sum(TfMRI(1:tr-1)) + 1; t1 = sum(TfMRI(1:tr));
    %Y2(t0:t1,:) = 0.1 * mvnrnd(zeros(TfMRI(tr),ndim), ...
    %    hmm_orig.HRF(tr).sigma.rate ./ hmm_orig.HRF(tr).sigma.shape);
    for n=1:ndim
        G = zeros(TfMRI(tr),1);
        for t=t0:t1
            G(t-t0+1) = sum(X2(hmm0.train.I2(t,:),n)' .* hmm0.train.meanH,2)';
        end
        Y2(t0:t1,n) =  Y2(t0:t1,n) + G;
    end
end
data2.Y = Y2; 
data2.T = data.T;
data2.Hz = data.Hz;

Xhat = obsinit(data2,hmm0);
Xhat2 = obsinit2(data2,hmm0);

%Xhat2.mu = (Xhat2.mu ./ repmat(std(Xhat2.mu),size(Xhat2.mu,1),1)) .* repmat(std(Xhat.mu),size(Xhat2.mu,1),1);

plot([X2(29:1028,1) Xhat.mu(29:1028,1) Xhat2.mu(29:1028,1)]) % Y2(1:1000,1)])

plot([(X2(29:1028,1)-Xhat.mu(29:1028,1)) (X2(29:1028,1)-Xhat2.mu(29:1028,1))]) % Y2(1:1000,1)])
