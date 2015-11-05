function hmm=simmodel(TfMRI,Tsignal,p,ndim,K,Hz,covtype,HRFconstant,StatePermanency,noiseFactor,train)
% Hz is frequency of the latent signal;
% train, if provided, must contain H, meanB and covB

N = length(TfMRI);

% Basis functions
if nargin<12
    [train.H,~,train.meanB,train.covB] = HRFbasis(p,Hz);
end
hmm.train.H = train.H;
hmm.train.covtype = covtype;
    
% transition matrix and initial probabilities
hmm.P = ones(K,K) + StatePermanency * eye(K); %rand(K,K);
for j=1:K,
    hmm.P(j,:)=hmm.P(j,:)./sum(hmm.P(j,:));
end;
hmm.Pi = ones(1,K); %rand(1,K);
hmm.Pi=hmm.Pi./sum(hmm.Pi);

% HRFs
for tr=1:N
    hmm.HRF(tr).B.mu = zeros(p+1,ndim);
    hmm.HRF(tr).B.S = zeros(p+1,p+1,ndim);
    hmm.HRF(tr).alpha.shape = zeros(1,ndim);
    hmm.HRF(tr).alpha.rate = zeros(1,ndim);
    hmm.HRF(tr).sigma.shape = zeros(1,ndim);
    hmm.HRF(tr).sigma.rate = zeros(1,ndim);    
    for n=1:ndim
        if HRFconstant && tr>1
            hmm.HRF(tr).alpha.shape(n) = hmm.HRF(1).alpha.shape(n);
            hmm.HRF(tr).alpha.rate(n) = hmm.HRF(1).alpha.rate(n);
            hmm.HRF(tr).sigma.shape(n) = hmm.HRF(1).sigma.shape(n);
            hmm.HRF(tr).sigma.rate(n) = hmm.HRF(1).sigma.rate(n);
            hmm.HRF(tr).B.mu(:,n) = hmm.HRF(1).B.mu(:,n);
            hmm.HRF(tr).B.S(:,:,n) = hmm.HRF(1).B.S(:,:,n);
        else
            hmm.HRF(tr).alpha.shape(n) = 0.5 * p;
            hmm.HRF(tr).alpha.rate(n) = hmm.HRF(tr).alpha.shape(n); %hmm.HRF(tr).alpha.shape(n) * 0.25 * rand;
            hmm.HRF(tr).sigma.shape(n) = 0.5 * Tsignal(tr);
            hmm.HRF(tr).sigma.rate(n) = hmm.HRF(tr).sigma.shape(n) * noiseFactor(1) * rand;   
            hmm.HRF(tr).B.mu(:,n) =  ...
                mvnrnd(train.meanB',(hmm.HRF(tr).alpha.rate(n)/hmm.HRF(tr).alpha.shape(n)) * train.covB)';            
            %hmm.HRF(tr).B.S(:,:,n) = (hmm.HRF(tr).alpha.rate(n)/hmm.HRF(tr).alpha.shape(n)) * eye(p);
        end
    end
end

% Observation model
for k=1:K
    hmm.state(k) = struct('Omega',[],'Mean',[]);
    hmm.state(k).Mean.mu = randn(1,ndim);
    hmm.state(k).Mean.S = 0.001 * eye(ndim);
    if strcmp(hmm.train.covtype,'diag')
        hmm.state(k).Omega.shape = 0.5 * TfMRI(tr);
        hmm.state(k).Omega.rate = hmm.state(k).Omega.shape * randn(1,ndim).^2;
        hmm.state(k).Omega.irate = 1 ./ (hmm.state(k).Omega.rate);
    else
        hmm.state(k).Omega.shape = TfMRI(tr);
        rM = noiseFactor(2) * randn(ndim); rM = rM' * rM + 2 * eye(ndim); 
        hmm.state(k).Omega.rate = hmm.state(k).Omega.shape * rM;
        hmm.state(k).Omega.irate = inv(hmm.state(k).Omega.rate);
    end
end