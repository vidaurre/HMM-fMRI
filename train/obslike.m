function LL = obslike (hmm,X)
%
% Evaluate likelihood of data given observation model
%
% INPUT
% X          N by ndim latent signal
% hmm        hmm data structure
%
% OUTPUT
% LL          Likelihood of N data points
%
% Author: Diego Vidaurre, OHBA, University of Oxford
 
[T,ndim] = size(X.mu);
K = hmm.K;
 
LL=zeros(T,K);  
ltpi= ndim/2 * log(2*pi);

for k=1:K
    hs=hmm.state(k);
    
    switch hmm.train.covtype,
        case 'diag'
            ldetWishB = 0.5*sum(log(hs.Omega.rate));
            PsiWish_alphasum = 0.5*ndim*psi(hs.Omega.shape);  
            C = hs.Omega.shape ./ hs.Omega.rate;
        case 'full'
            ldetWishB = 0.5*hs.Omega.logdetrate;
            PsiWish_alphasum = 0;
            for n=1:ndim,
                PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hs.Omega.shape/2+0.5-n/2);
            end;
            C = hs.Omega.shape * hs.Omega.irate;
    end;
    
    d = X.mu - repmat(hs.Mean.mu,size(X.mu,1),1);
    if strcmp(hmm.train.covtype,'diag')
        Cd =  repmat(C',1,T) .* d';
    else
        Cd = C * d';
    end

    dist=zeros(T,1);
    for n=1:ndim
        dist=dist-0.5*d(:,n).*Cd(n,:)';
    end
       
    % uncertainty in the mean
    NormWishtrace = zeros(T,1);
    NormWishtrace(:) = 0.5 * sum(sum( C .* hs.Mean.S ));
    % uncertainty in X
    for t=1:T
        ind = (1:ndim) + ndim*(t-1);
        if strcmp(hmm.train.covtype,'diag')
            NormWishtrace(t) = NormWishtrace(t) + 0.5 * sum( C' .*  diag(X.S{1}(ind,ind)) );
        else
            NormWishtrace(t) = NormWishtrace(t) + 0.5 * sum(sum( C' .* X.S{1}(ind,ind) ));
        end
    end
    
    LL(:,k)= -ltpi - ldetWishB + PsiWish_alphasum + dist - NormWishtrace; 
end;
LL=exp(LL);
