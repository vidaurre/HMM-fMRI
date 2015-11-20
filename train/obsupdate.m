function hmm = obsupdate (hmm,Gamma,X,T)
%
% Update observation model
%
% INPUT
% hmm           hmm data structure
% Gamma         p(state given X,Y)
% X             latent signal
% T             number of time points for each latent time series
%
% OUTPUT
% hmm           estimated HMMMAR model
%
% Author: Diego Vidaurre, OHBA, University of Oxford

ndim = size(X.mu,2); K=length(hmm.state);
updateXindexes = hmm.train.updateXindexes;
Gammasum = sum(Gamma);
T = T - sum(abs(hmm.train.cutoff));

% Mean
for k=1:K,
    muk = Gamma(:,k)' * X.mu(updateXindexes,:);
    if strcmp(hmm.train.covtype,'diag')
        omega = hmm.state(k).Omega.shape ./ hmm.state(k).Omega.rate; % prec
        hmm.state(k).Mean.S = 1 ./ (Gammasum(k) * omega + hmm.state(k).prior.Mean.iS);
        hmm.state(k).Mean.mu = hmm.state(k).Mean.S .*  ...
            ( omega .* muk + hmm.state(k).prior.Mean.iSmu);
        hmm.state(k).Mean.logdetS = sum(log(hmm.state(k).Mean.S));
    else % full
        omega = hmm.state(k).Omega.irate * hmm.state(k).Omega.shape; % prec
        hmm.state(k).Mean.S = inv(Gammasum(k) * omega + diag(hmm.state(k).prior.Mean.iS));
        hmm.state(k).Mean.mu = hmm.state(k).Mean.S *  ...
            ( omega * muk' + hmm.state(k).prior.Mean.iSmu');
        hmm.state(k).Mean.mu = hmm.state(k).Mean.mu';
        hmm.state(k).Mean.logdetS = logdet(hmm.state(k).Mean.S,'chol');
    end
end

% Covariance % the error has to be either here or somewhere where Omega pops in
if strcmp(hmm.train.covtype,'diag')
    S = zeros(length(updateXindexes),ndim);
else
    S = zeros(length(updateXindexes),ndim,ndim);
end
pos = 1;
% making S - the covariance matrix limited to intranode interactions
for tr=1:length(T) 
    for t=1:T(tr)
        ind = (1:ndim) + ndim*(t-1);
        if strcmp(hmm.train.covtype,'diag')
            S(pos,:) = diag(X.S{tr}(ind,ind))';
        else
            S(pos,:,:) = X.S{tr}(ind,ind);
        end
        pos = pos + 1;
    end
end
for k=1:K
    dist = X.mu(updateXindexes,:) - repmat(hmm.state(k).Mean.mu,length(updateXindexes),1);
    if strcmp(hmm.train.covtype,'diag')
        S1 = sum(repmat(Gamma(:,k),1,ndim) .* S,1);
        S2 = Gammasum(k) * hmm.state(k).Mean.S;
        hmm.state(k).Omega.rate = hmm.state(k).prior.Omega.rate + ...
                0.5 * sum( repmat(Gamma(:,k),1,ndim) .* (dist.^2) ) + 0.5 * (S1 + S2);
        hmm.state(k).Omega.irate = 1 ./ hmm.state(k).Omega.rate;
        hmm.state(k).Omega.shape = hmm.state(k).prior.Omega.shape + 0.5 * Gammasum(k);
    else % full
        M = (dist .* repmat(Gamma(:,k),1,ndim))' * dist;
        S1 = permute(sum(repmat(Gamma(:,k),[1,ndim,ndim]) .* S,1),[2 3 1]); % uncertainty in X
        S2 = Gammasum(k) * hmm.state(k).Mean.S; % uncertainty in the mean
        hmm.state(k).Omega.rate = hmm.state(k).prior.Omega.rate + (M + S1 + S2);
        hmm.state(k).Omega.irate = inv(hmm.state(k).Omega.rate);
        hmm.state(k).Omega.shape = hmm.state(k).prior.Omega.shape + Gammasum(k);
        hmm.state(k).Omega.logdetrate = logdet(hmm.state(k).Omega.rate,'chol');
    end
end

end
