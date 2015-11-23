function [FrEn] = evalfreeenergy (data,T,hmm,Gamma,Xi,X)
% Computes the Free Energy of an HMM depending on observation model
%
% INPUT
% X            observations
% T            length of series
% Gamma        probability of states conditioned on data
% Xi           joint probability of past and future states conditioned on data
% hmm          data structure
% residuals    in case we train on residuals, the value of those.
%
% OUTPUT
% FrEn         value of the variational free energy, separated in the
%               different terms
%
% Author: Diego Vidaurre, OHBA, University of Oxford


ndim = size(X.mu,2); K = hmm.K; 
N = length(T);
cutoff = hmm.train.cutoff; scutoff = sum(abs(cutoff));
updateXindexes = hmm.train.updateXindexes; 
Gammasum=sum(Gamma,1);

% Entropy of hidden states
% EntrGamma0=0;
% for tr=1:length(T);
%     t = sum(T(1:tr-1)) - (tr-1)*scutoff + 1;
%     logGamma = log(Gamma(t,:));
%     logGamma(isinf(-logGamma)) = log(realmin);
%     EntrGamma0 = EntrGamma0 - sum(Gamma(t,:).*logGamma);
%     tt = t+1:(sum(T(1:tr))-tr*scutoff);    
%     Gammatt = Gamma(tt,:);
%     logGamma = log(Gammatt);
%     logGamma(isinf(-logGamma)) = log(realmin);
%     EntrGamma0 = EntrGamma0 + sum(Gammatt(:).*logGamma(:));
% end
% sXi = Xi(:); 
% logsXi = log(sXi);
% logsXi(isinf(-logsXi)) = log(realmin);
% EntrGamma0 = EntrGamma0 - sum(sXi .* logsXi);

EntrGamma=0;
for tr=1:length(T);
    t = sum(T(1:tr-1)) - (tr-1)*scutoff + 1;
    Gamma_nz = Gamma(t,:); Gamma_nz(Gamma_nz==0) = realmin;
    EntrGamma = EntrGamma - sum(Gamma_nz .* log(Gamma_nz));
    ttXi = (sum(T(1:tr-1)) - (tr-1)*(scutoff+1) + 1) : ((sum(T(1:tr)) - tr*(scutoff+1)));
    ttGamma = t:(sum(T(1:tr))-tr*scutoff-1);   
    for s=1:length(ttXi)
        Xi_nz = Xi(ttXi(s),:,:); Xi_nz(Xi_nz==0) = realmin;
        Gamma_nz = Gamma(ttGamma(s),:); Gamma_nz(Gamma_nz==0) = realmin;
        EntrGamma = EntrGamma - sum(sum(Xi_nz .* log( Xi_nz ) )); 
        EntrGamma = EntrGamma + sum(Gamma_nz .* log( Gamma_nz )); 
    end
end

% EntrGamma2=0;
% for tr=1:length(T);
%     t = sum(T(1:tr-1)) - (tr-1)*scutoff + 1;
%     Gamma_nz = Gamma(t,:); Gamma_nz(Gamma_nz==0) = realmin;
%     EntrGamma2 = EntrGamma2 - sum(Gamma_nz .* log(Gamma_nz));
%     ttXi = (sum(T(1:tr-1)) - (tr-1)*(scutoff+1) + 1) : ((sum(T(1:tr)) - tr*(scutoff+1)));
%     Xi_tr = Xi(ttXi,:,:); Xi_tr(Xi_tr==0) = realmin;
%     Psi=zeros(size(Xi_tr));                    % P(S_t|S_t-1)
%     for k=1:K,
%         sXi=sum(squeeze(Xi_tr(:,:,k)),2);
%         Psi(:,:,k)=Xi_tr(:,:,k)./repmat(sXi,1,K);
%     end
%     EntrGamma2=EntrGamma2+sum(Xi_tr(:).*log(Psi(:)),1);
% end
% EntrGamma2 = - EntrGamma2;

% avLL and KL-divergence for hidden state  % NOTHING ON P AND PI??????
KLdiv=dirichlet_kl(hmm.Dir_alpha,hmm.prior.Dir_alpha);
% for l=1:K, % KL-divergence for transition prob
%     KLdiv = [KLdiv dirichlet_kl(hmm.Dir2d_alpha(l,:),hmm.prior.Dir2d_alpha(l,:))];
% end
KLdiv = [KLdiv dirichlet_kl(hmm.Dir2d_alpha(:)',hmm.prior.Dir2d_alpha(:)')];
avLLGamma = 0;
tt = zeros(length(T),1);
for tr=1:length(T);
    tt(tr) = sum(T(1:tr-1)) - (tr-1)*scutoff + 1;
end
PsiDir_alphasum = psi(sum(hmm.Dir_alpha));
for l=1:K, % avLL initial state 
    avLLGamma = avLLGamma + sum(Gamma(tt,l)) * (psi(hmm.Dir_alpha(l)) - PsiDir_alphasum) ;
end    
% avLL remaining states  
PsiDir2d_alphasum = psi(sum(hmm.Dir2d_alpha(:)));
for l=1:K, 
    for k=1:K,
        avLLGamma = avLLGamma + sum(Xi(:,l,k)) * (psi(hmm.Dir2d_alpha(l,k)) - PsiDir2d_alphasum);
    end;
end;

% Entropy of latent signal
EntrX = 0;
for tr=1:length(T)
    try
        EntrX = EntrX + 0.5 * logdet(X.S{tr},'chol') + 0.5 * size(X.S{tr},1) * (1 + log(2*pi));
    catch exception
        warning(strcat('Covariance of the latent signal for trial ',num2str(tr),...
            ' is close to not be positive definite, so I cannot use the choleski factorization'))
        EntrX = EntrX + 0.5 * logdet(X.S{tr}) + 0.5 * size(X.S{tr},1) * (1 + log(2*pi));
    end
end

ltpi = ndim/2*log(2*pi); % - ndim/2;
OmegaKL = 0; meanKL = 0;
avLLX = 0;

for k=1:K,
    
    hs = hmm.state(k);		% for ease of referencing
    
    if strcmp(hmm.train.covtype,'diag')
        ldetWishB=0;
        PsiWish_alphasum=0;
        for n=1:ndim,
            ldetWishB=ldetWishB+0.5*log(hs.Omega.rate(n));
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hs.Omega.shape);  
        end;
        C = hs.Omega.shape ./ hs.Omega.rate;
    else
        ldetWishB=0.5*hs.Omega.logdetrate;
        PsiWish_alphasum=0;
        for n=1:ndim,
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hs.Omega.shape/2+0.5-n/2); 
        end;
        C = hs.Omega.shape * hs.Omega.irate;
    end
    avLLX = avLLX + Gammasum(k) * (-ltpi-ldetWishB+PsiWish_alphasum);
    
    % average log likelihood for latent signal
    d = X.mu(updateXindexes,:) - repmat(hs.Mean.mu,length(updateXindexes),1);
    if strcmp(hmm.train.covtype,'diag')
        Cd =  repmat(C',1,length(updateXindexes)) .* d';
    else
        Cd = C * d';
    end
    dist = zeros(length(updateXindexes),1);
    for n = 1:ndim,
        dist = dist-0.5*d(:,n).*Cd(n,:)';
    end
    % uncertainty in the mean
    NormWishtrace = zeros(length(updateXindexes),1);
    NormWishtrace(:) = 0.5 * sum(sum( C .* hs.Mean.S ));
    % uncertainty in X
    pos = 1;
    for tr=1:length(T)
        Ttr = T(tr)-scutoff;
        for t=1:Ttr
            ind = (1:ndim) + ndim*(t-1);
            if strcmp(hmm.train.covtype,'diag')
                NormWishtrace(pos) = NormWishtrace(pos) + 0.5 * sum( C' .*  diag(X.S{tr}(ind,ind)) );
            else
                NormWishtrace(pos) = NormWishtrace(pos) + 0.5 * sum(sum( C' .* X.S{tr}(ind,ind) ));
            end
            pos = pos + 1;
        end
    end
        
    avLLX = avLLX + sum(Gamma(:,k).*(dist - NormWishtrace));

    % KL for obs models 
    switch hmm.train.covtype
        case 'diag'
            OmegaKL = 0;
            for n=1:ndim
                OmegaKL = OmegaKL + gamma_kl(hs.Omega.shape,hs.prior.Omega.shape, ...
                    hs.Omega.rate(n),hs.prior.Omega.rate(n));
            end;
            meanKL = meanKL + gauss_kl(hs.Mean.mu,hs.prior.Mean.mu,diag(hs.Mean.S),diag(hs.prior.Mean.S),...
                hs.Mean.logdetS,hs.prior.Mean.logdetS,diag(hs.prior.Mean.iS));
        case 'full'
            OmegaKL = wishart_kl(hs.Omega.rate,hs.prior.Omega.rate, hs.Omega.shape,hs.prior.Omega.shape,...
                hs.Omega.logdetrate,hs.prior.Omega.logdetrate);
            meanKL = meanKL + gauss_kl(hs.Mean.mu,hs.prior.Mean.mu,hs.Mean.S,diag(hs.prior.Mean.S),...
                hs.Mean.logdetS,hs.prior.Mean.logdetS,diag(hs.prior.Mean.iS));
    end

    KLdiv = [KLdiv OmegaKL meanKL];   
    
end;

% average log likelihood for y

avLLY = 0;

for tr=1:N
    
    ldetWishB = 0;
    PsiWish_alphasum = 0;
    for n=1:ndim,
        ldetWishB = ldetWishB+0.5*log(hmm.HRF(tr).sigma.rate(n));
        PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.HRF(tr).sigma.shape(n)); 
    end;
    
    avLLY = avLLY + data.T(tr)*(-ltpi-ldetWishB+PsiWish_alphasum);
    %t0 = sum(data.T(1:tr-1)); t1 = sum(data.T(1:tr));
    y = gety(data,T,tr,hmm,X);
    resp = zeros(data.T(tr),ndim);
    dist=zeros(data.T(tr),1);
    NormWishtrace1 = zeros(data.T(tr),1);
    NormWishtrace2 = 0;
    C = hmm.HRF(tr).sigma.shape ./ hmm.HRF(tr).sigma.rate;
    for n=1:ndim
       [G,V] = buildGV(data.T,T,hmm,X,n,tr);
       resp(:,n) = G * hmm.HRF(tr).B.mu(:,n);
       NormWishtrace1 = NormWishtrace1 + 0.5 * C(n) *  ...
           sum( (G * hmm.HRF(tr).B.S(:,:,n)) .* G, 2);
       NormWishtrace2 = NormWishtrace2 + 0.5 * C(n) * sum(sum(V .* hmm.HRF(tr).B.S(:,:,n))) + ...
           0.5 * C(n) * sum(sum(V .* (hmm.HRF(tr).B.mu(:,n) * hmm.HRF(tr).B.mu(:,n)') )) ;
    end
    d = y - resp;
    Cd = repmat(C',1,data.T(tr)) .* d';  
    for n=1:ndim
        dist = dist - 0.5 * d(:,n) .* Cd(n,:)';
    end
    avLLY = avLLY + sum(dist - NormWishtrace1) - NormWishtrace2;
    
end

% KL for HRF models
sigmaKL = 0; alphaKL = 0; BKL = 0;
for tr=1:N
    for n=1:ndim
        sigmaKL = sigmaKL + gamma_kl(hmm.HRF(tr).sigma.shape(n),hmm.HRF(tr).prior.sigma.shape(n), ...
            hmm.HRF(tr).sigma.rate(n),hmm.HRF(tr).prior.sigma.rate(n));
        alphaKL = alphaKL + gamma_kl(hmm.HRF(tr).alpha.shape(n),hmm.HRF(tr).prior.alpha.shape(n), ...
            hmm.HRF(tr).alpha.rate(n),hmm.HRF(tr).prior.alpha.rate(n));
        alph = hmm.HRF(tr).alpha.rate(n) / hmm.HRF(tr).alpha.shape(n);
        BKL = BKL + gauss_kl(hmm.HRF(tr).B.mu(:,n), hmm.HRF(tr).prior.B.mu,  ...
            hmm.HRF(tr).B.S(:,:,n), alph * hmm.HRF(tr).prior.B.S);
    end
end
KLdiv=[KLdiv sigmaKL alphaKL BKL]; 
if any(isnan(KLdiv)), 
    warning('There are NaN in the free energy calculation')
    keyboard; 
end

FrEn=[-EntrGamma -EntrX -avLLGamma -avLLX -avLLY +KLdiv];

fprintf('-EntrGamma=%g, -EntrX=%g, -avLLGamma=%g, -avLLX=%g -avLLY=%g KLdiv=%g \n',...
   sum(-EntrGamma),sum(-EntrX),sum(-avLLGamma),sum(-avLLX),sum(-avLLY),sum(KLdiv) )


