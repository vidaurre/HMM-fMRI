function [hmm,X,Gamma] = hmminit(data,T,hmm)
%
% Initialise observation model in HMM
%
% INPUT
% data          observations - a struct with X (time series) and C (classes)
% T             Number of time points for each latent time series
% hmm           hmm data structure
%
% OUTPUT
% hmm           initial HMM model
% X             initial latent series
% Gamma         estimated state time courses
%
% Author: Diego Vidaurre, OHBA, University of Oxford

ndim = size(data.Y,2); K = hmm.train.K;
hmm = initpriors(data,hmm); % priors
hmm = initpost(data.T,hmm,ndim); % posteriori
[X,updateXindexes] = obsinit(data,hmm); % init latent signal
hmm.train.updateXindexes = updateXindexes;
%hmm.train.Yboundary = Yboundary(data,hmm,X); % because the start and end of X is not updated, Y needs to be adjusted (pending)
scutoff = sum(abs(hmm.train.cutoff));
hmm = HRFupdate(data,T,hmm,X); % HRF model
X = obsinference(data,T,hmm,(1/K) * ones(sum(T)-length(T)*scutoff,K),X); % infer the latent signal only based on Y
if isempty(hmm.train.Gamma) % states
    if strcmp(hmm.train.inittype,'GMM'), 
        Gamma = gmm_init(X.mu(updateXindexes,:),T-scutoff,hmm.train); 
    elseif strcmp(hmm.train.inittype,'HMM-fMRI'), 
        Gamma = hmmfmri_init(data,hmm.train);
    else % options.inittype=='random'
        Gamma = initGamma_random(T-scutoff,hmm.train.K,hmm.train.DirichletDiag);
    end
else
    Gamma = hmm.train.Gamma;
    if size(Gamma,1)==sum(T-scutoff), 
        error('Dimensions of starting Gamma are incorrect')
    end
end; 
hmm.train = rmfield(hmm.train,'Gamma');
hmm.train = rmfield(hmm.train,{'meanB','covB'});

end


function hmm = initpriors(data,hmm)
% define priors
ndim = size(data.Y,2);
r = range(data.Y);   
r2 = r.^2;  

for k=1:hmm.K,
    hmm.state(k).prior=struct('Omega',[],'Mean',[]);
    hmm.state(k).prior.Mean = struct('mu',[],'iS',[]);
    hmm.state(k).prior.Mean.mu = zeros(1,ndim);
    hmm.state(k).prior.Mean.S = r2;
    hmm.state(k).prior.Mean.iS = 1./r2;
    hmm.state(k).prior.Mean.iSmu = hmm.state(k).prior.Mean.iS .* hmm.state(k).prior.Mean.mu;
    if strcmp(hmm.train.covtype,'full')
        hmm.state(k).prior.Omega.shape = ndim+0.1-1;
        hmm.state(k).prior.Omega.rate = diag(r);
        hmm.state(k).prior.Omega.irate = diag(1 ./ r);
    else      
        hmm.state(k).prior.Omega.shape = 0.5 * (ndim+0.1-1);
        hmm.state(k).prior.Omega.rate = 0.5 * r;
        hmm.state(k).prior.Omega.irate = 1 ./ hmm.state(k).prior.Omega.rate;
    end
end;

icovB = inv(hmm.train.beta * hmm.train.covB);
for tr=1:length(data.T)
    hmm.HRF(tr).prior.B.mu = hmm.train.meanB;
    hmm.HRF(tr).prior.B.S = hmm.train.beta * hmm.train.covB;
    hmm.HRF(tr).prior.B.iS = icovB;
    hmm.HRF(tr).prior.alpha.shape = 0.1*ones(1,ndim); 
    hmm.HRF(tr).prior.alpha.rate = 0.1*ones(1,ndim);   
    hmm.HRF(tr).prior.sigma.shape = 0.1*ones(1,ndim);
    hmm.HRF(tr).prior.sigma.rate = 0.1*ones(1,ndim);
end

end


function hmm = initpost(T,hmm,ndim)

%updateXindexes = hmm.train.updateXindexes;
for tr=1:length(T)
    hmm.HRF(tr).alpha.shape = hmm.HRF(tr).prior.alpha.shape;
    hmm.HRF(tr).alpha.rate = hmm.HRF(tr).prior.alpha.rate; 
    hmm.HRF(tr).sigma.shape = hmm.HRF(tr).prior.sigma.shape;
    hmm.HRF(tr).sigma.rate = hmm.HRF(tr).prior.sigma.rate;
end
for k=1:hmm.K,
    %muk = Gamma(:,k)' * X.mu(updateXindexes,:);
    %hmm.state(k).Mean.mu = muk;
    %dist = X.mu(updateXindexes,:) - repmat(muk,size(X.mu(updateXindexes,:),1),1);
    hmm.state(k).Mean.mu = zeros(1,ndim);
    if strcmp(hmm.train.covtype,'full')
        hmm.state(k).Omega.rate = hmm.state(k).prior.Omega.rate; % + ...
        %    (dist .* repmat(Gamma(:,k),1,ndim))' * dist;
        hmm.state(k).Omega.irate = inv(hmm.state(k).Omega.rate);
        hmm.state(k).Omega.shape = hmm.state(k).prior.Omega.shape; % + sum(Gamma(:,k));
    else
        hmm.state(k).Omega.rate = hmm.state(k).prior.Omega.rate; % + ...
        %    0.5 * sum( repmat(Gamma(:,k),1,ndim) .* dist.^2  );
        hmm.state(k).Omega.shape = hmm.state(k).prior.Omega.shape; % + sum(Gamma(:,k));
    end
end

end





