function [FrEn] = evalfreeenergy0 (data,T,hmm,Gamma,Xi,X)
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


ndim = size(X.mu,2); K=hmm.K; 
sumT = sum(T); N = length(T);

Gammasum=sum(Gamma,1);

% Entropy of hidden states
EntrGamma=0;
% for in=1:length(T);
%     j = sum(T(1:in-1)) + 1;
%     logGamma = log(Gamma(j,:));
%     logGamma(isinf(-logGamma)) = log(realmin);
%     EntrGamma = EntrGamma - sum(Gamma(j,:).*logGamma);
%     jj = j+1:j+T(in)-1;
%     Gammajj = Gamma(jj,:);
%     logGamma = log(Gammajj);
%     logGamma(isinf(-logGamma)) = log(realmin);
%     EntrGamma = EntrGamma + sum(Gammajj(:).*logGamma(:));
% end
% sXi = Xi(:); 
% logsXi = log(sXi);
% logsXi(isinf(-logsXi)) = log(realmin);
% EntrGamma = EntrGamma - sum(sXi .* logsXi);

% Entr=0;
% Xi(Xi==0)=realmin;				% avoid log(0)
% Psi=zeros(size(Xi));			% P(S_t|S_t-1)
% for k=1:K,
%     sXi=sum(squeeze(Xi(:,:,k)),2);
%     Psi(:,:,k)=Xi(:,:,k)./repmat(sXi,1,K);
% end;
% Psi(Psi==0)=realmin;				% avoid log(0)
% Entr=Entr+sum(Xi(:).*log(Psi(:)),1);
% Entr=Entr+sum(Xi(:).*log(Psi(:) ./  Xi(:)),1);	% entropy of hidden states

avLL = 0; 
KLdiv= 0;

% avLL and KL-divergence for hidden state  
% KLdiv=dirichlet_kl(hmm.Dir_alpha,hmm.prior.Dir_alpha);
% avLL = -length(T) * psi(sum(hmm.Dir_alpha));
% jj = zeros(length(T),1);
% for in=1:length(T);
%     jj(in) = sum(T(1:in-1)) + 1;
% end
% for l=1:K,
%     % KL-divergence for transition prob
%     KLdiv=[KLdiv dirichlet_kl(hmm.Dir2d_alpha(l,:),hmm.prior.Dir2d_alpha(l,:))];
%     % avLL initial state  
%     avLL = avLL + sum(Gamma(jj,l)) * psi(hmm.Dir_alpha(l));
% end    
% % avLL remaining states  
% for k=1:K,
%     sXi = Xi(:,:,k); sXi = sum(sXi(:));  
%     avLL = avLL - sXi * psi(sum(hmm.Dir2d_alpha(:,k)));
%     for l=1:K,
%         avLL = avLL + sum(Xi(:,l,k)) * psi(hmm.Dir2d_alpha(l,k));
%     end;
% end;

% Entropy of latent signal
EntrX = 0;
% for t=1:sumT
%     EntrX = EntrX + logdet(permute(X.S(t,:,:),[2 3 1]));
% end
% EntrX = 0.5 * EntrX + 0.5 * ndim * (1 + log(2*pi));

ltpi = ndim/2*log(2*pi); % - ndim/2;
OmegaKL = 0; meanKL = 0;
for k=1:K,
    
    hs=hmm.state(k);		% for ease of referencing
    
    if strcmp(hmm.train.covtype,'diag')
        ldetWishB=0;
        PsiWish_alphasum=0;
        for n=1:ndim,
            ldetWishB=ldetWishB+0.5*log(hs.Omega.rate(n));
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hs.Omega.shape/2);
        end;
        C = hs.Omega.shape ./ hs.Omega.rate;
        avLL=avLL+ Gammasum(k)*(-ltpi-ldetWishB+PsiWish_alphasum);
    else
        ldetWishB=0.5*logdet(hs.Omega.rate);
        PsiWish_alphasum=0;
        for n=1:ndim,
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hs.Omega.shape/2+0.5-n/2);
        end;
        C = hs.Omega.shape * hs.Omega.irate;
        avLL =avLL + Gammasum(k) * (-ltpi-ldetWishB+PsiWish_alphasum);
    end;
    
    % average log likelihood for latent signal
    d = X.mu - repmat(hs.Mean.mu,size(X.mu,1),1);
    if strcmp(hmm.train.covtype,'diag')
        Cd =  repmat(C',1,sumT) .* d';
    else
        Cd = C * d';
    end
    dist=zeros(sumT,1);
    for n=1:ndim,
        dist=dist-0.5*d(:,n).*Cd(n,:)';
    end
    if strcmp(hmm.train.covtype,'diag')
        NormWishtrace = 0.5 * sum(repmat(C,sumT,1) .* (repmat(hs.Mean.S,sumT,1) + X.S) , 2);
    else
        NormWishtrace = zeros(sumT,1);
        for n1=1:ndim
            for n2=1:ndim
                NormWishtrace = NormWishtrace + 0.5 * C(n1,n2) * ( hs.Mean.S(n1,n2) + X.S(:,n1,n2));
            end
        end
    end
    avLL = avLL + sum(Gamma(:,k).*(dist - NormWishtrace));

    % KL for obs models 
    switch hmm.train.covtype
        case 'diag'
            OmegaKL = 0;
            for n=1:ndim
                OmegaKL = OmegaKL + gamma_kl(hs.Omega.shape,hs.prior.Omega.shape, ...
                    hs.Omega.rate(n),hs.prior.Omega.rate(n));
            end;
            meanKL = meanKL + gauss_kl(hs.Mean.mu,hs.prior.Mean.mu,diag(hs.Mean.S),diag(hs.prior.Mean.S));
        case 'full'
            OmegaKL = wishart_kl(hs.Omega.rate,hs.prior.Omega.rate, ...
                hs.Omega.shape,hs.prior.Omega.shape);
            meanKL = meanKL + gauss_kl(hs.Mean.mu,hs.prior.Mean.mu,hs.Mean.S,diag(hs.prior.Mean.S));
    end

    KLdiv=[KLdiv OmegaKL meanKL];   
    
end;

% average log likelihood for y
for tr=1:N
    t0 = sum(data.T(1:tr-1)); t1 = sum(data.T(1:tr));
    resp = zeros(data.T(tr),ndim);
    dist=zeros(data.T(tr),1);
    NormWishtrace = zeros(data.T(tr),1);
    for n=1:ndim
       [G,V] = buildGV(data.T(tr),hmm,X,n);
       resp(:,n) = G * hmm.HRF(tr).B.mu(:,n);
       NormWishtrace = NormWishtrace + 0.5 * C(n) *  ...
           ( sum( (G * hmm.HRF(tr).B.S(:,:,n)) .* G, 2) + ...
           sum(sum(V .* hmm.HRF(tr).B.S(:,:,n))) );
    end
    C =  hmm.HRF(tr).sigma.shape ./ hmm.HRF(tr).sigma.rate;
    d = data.Y(t0+1:t1,:) - resp;
    Cd = repmat(C',1,data.T(tr)) .* d';  
    for n=1:ndim
        dist = dist - 0.5 * d(:,n) .* Cd(n,:)';
    end
    avLL = avLL + sum(dist - NormWishtrace);
end

% KL for HRF models
sigmaKL = 0; alphaKL = 0; BKL = 0;
for tr=1:N
    for n=1:ndim
        sigmaKL = sigmaKL + gamma_kl(hmm.HRF(tr).sigma.shape(n),hmm.HRF(tr).prior.sigma.shape(n), ...
            hmm.HRF(tr).sigma.rate(n),hmm.HRF(tr).prior.sigma.rate(n));
        alphaKL = alphaKL + gamma_kl(hmm.HRF(tr).alpha.rate(n),hmm.HRF(tr).prior.alpha.rate(n), ...
            hmm.HRF(tr).alpha.shape(n),hmm.HRF(tr).prior.alpha.shape(n));
        BKL = BKL + gauss_kl(hmm.HRF(tr).B.mu(:,n), hmm.HRF(tr).prior.B.mu,  ...
            hmm.HRF(tr).B.S(:,:,n), (hmm.HRF(tr).alpha.rate(n) / hmm.HRF(tr).alpha.shape(n)) * ...
            hmm.HRF(tr).prior.B.S);
    end
end
KLdiv=[KLdiv sigmaKL alphaKL BKL]; 
if any(isnan(KLdiv)), keyboard; end

FrEn=[EntrGamma EntrX -avLL +KLdiv];
