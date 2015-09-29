function hmm = obsupdate (hmm,Gamma,X)
%
% Update observation model
%
% INPUT
% hmm           hmm data structure
% Gamma         p(state given X,Y)
% X             latent signal
%
% OUTPUT
% hmm           estimated HMMMAR model
%
% Author: Diego Vidaurre, OHBA, University of Oxford

ndim = size(X.mu,2); K=length(hmm.state);
updateXindexes = hmm.train.updateXindexes;
Gammasum = sum(Gamma);

% Mean
for k=1:K,
    muk = Gamma(:,k)' * X.mu(updateXindexes,:);
    if strcmp(hmm.train.covtype,'diag')
        omega = hmm.state(k).Omega.shape ./ hmm.state(k).Omega.rate; % prec
        hmm.state(k).Mean.S = 1 ./ (Gammasum(k) * omega + hmm.state(k).prior.Mean.iS);
        hmm.state(k).Mean.mu = hmm.state(k).Mean.S .*  ...
            ( omega .* muk + hmm.state(k).prior.Mean.iSmu);
    else % full
        omega = hmm.state(k).Omega.irate * hmm.state(k).Omega.shape; % prec
        hmm.state(k).Mean.S = inv(Gammasum(k) * omega + diag(hmm.state(k).prior.Mean.iS));
        hmm.state(k).Mean.mu = hmm.state(k).Mean.S *  ...
            ( omega * muk' + hmm.state(k).prior.Mean.iSmu');
        hmm.state(k).Mean.mu = hmm.state(k).Mean.mu';
    end
end

% Covariance
for k=1:K
    dist = X.mu(updateXindexes,:) - repmat(hmm.state(k).Mean.mu,length(updateXindexes),1);
    if strcmp(hmm.train.covtype,'diag')
        hmm.state(k).Omega.rate = hmm.state(k).prior.Omega.rate + ...    
            0.5 * sum( repmat(Gamma(:,k),1,ndim) .* ( dist.^2 + X.S(updateXindexes,:)) ) + ...
            0.5 * Gammasum(k) * hmm.state(k).Mean.S ;
        hmm.state(k).Omega.irate = 1 ./ hmm.state(k).Omega.rate;
        hmm.state(k).Omega.shape = hmm.state(k).prior.Omega.shape + 0.5 * Gammasum(k);
    else % full
        M = (dist .* repmat(Gamma(:,k),1,ndim))' * dist;
        S1 = permute(sum(repmat(Gamma(:,k),[1,ndim,ndim]) .* X.S(updateXindexes,:,:),1),[2 3 1]);
        S2 = Gammasum(k) * hmm.state(k).Mean.S;
        hmm.state(k).Omega.rate = hmm.state(k).prior.Omega.rate + (M + S1 + S2);
        hmm.state(k).Omega.irate = inv(hmm.state(k).Omega.rate);
        hmm.state(k).Omega.shape = hmm.state(k).prior.Omega.shape + Gammasum(k);
    end
end

end
