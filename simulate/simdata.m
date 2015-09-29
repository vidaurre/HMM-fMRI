function [Y,X,T,Gamma]=simdata(hmm,Gamma,TfMRI,HzfMRI,Hzsignal,smooth_gamma)
%
% Simulate data from the HMM-fMRI
%
% INPUTS:
%
% hmm           hmm structure with options specified in hmm.train
% Gamma         Initial state courses (specify to [] is this is to be sampled too)
% HzfMRI        frequency of the fMRI signal 
% HzSignal      frequency of the latent signal 
% Tfmri         Number of time points for each time series (fmri space)
% smooth_gamma  Smoothing the generated state time courses? 
%
% OUTPUTS
% X             simulated observations  
% T             Number of time points for each time series (latent time series)
% Gamma         simulated  p(state | data)
%
% Author: Diego Vidaurre, OHBA, University of Oxford

N = length(TfMRI); K = length(hmm.state);
ndim = size(hmm.HRF(1).B.mu,2); 
if nargin<6, smooth_gamma=0; end

[p,L] = size(hmm.train.H);
[~,I2,T] = initindexes(TfMRI,L,HzfMRI,Hzsignal);
if ~isempty(Gamma) && size(Gamma,1)~=sum(T), 
    error('Hz and Gamma size are not consistent')
elseif isempty(Gamma), % Gamma is not provided, so we simulate it too
    for in=1:N
        Gammai = zeros(T(in),K);
        Gammai(1,:) = mnrnd(1,hmm.Pi);
        for t=2:T(in)
            Gammai(t,:) = mnrnd(1,hmm.P(Gammai(t-1,:)==1,:));
        end
        if smooth_gamma>0
            for k=1:K,
                Gammai(:,k) = smooth(Gammai(:,k),'lowess',smooth_gamma);
            end
        end
        Gamma = [ Gamma;  Gammai ./ repmat(sum(Gammai,2),1,K) ];
    end
end

% simulation of the latent signal
X = zeros(sum(T),ndim);
for k=1:K
    if strcmp(hmm.train.covtype,'diag')
        X = X + repmat(Gamma(:,k),1,ndim) .* mvnrnd(repmat(hmm.state(k).Mean.mu,sum(T),1),...
            diag(hmm.state(k).Omega.rate / hmm.state(k).Omega.shape));
    else
        X = X + repmat(Gamma(:,k),1,ndim) .* mvnrnd(repmat(hmm.state(k).Mean.mu,sum(T),1),...
            hmm.state(k).Omega.rate / hmm.state(k).Omega.shape);
    end
end
    
% simulation of the fmri signal
Y = zeros(sum(TfMRI),ndim);
for tr=1:N
    t0 = sum(TfMRI(1:tr-1)) + 1; t1 = sum(TfMRI(1:tr));
    Y(t0:t1,:) = mvnrnd(zeros(TfMRI(tr),ndim), ...
        hmm.HRF(tr).sigma.rate ./ hmm.HRF(tr).sigma.shape);
    for n=1:ndim
        G = zeros(TfMRI(tr),p);
        for t=t0:t1
            G(t-t0+1,:) = sum(repmat(X(I2(t,:),n)',p,1) .* hmm.train.H,2)';
        end
        Y(t0:t1,n) =  Y(t0:t1,n) + G * hmm.HRF(tr).B.mu(:,n);
    end
end

end

