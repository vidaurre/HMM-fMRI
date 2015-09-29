function [hmm]=hsupdate(hmm,T,Xi,Gamma)
%
% updates hidden state parameters of an HMM
%
% INPUT:
%
% hmm    single hmm data structure
% T      length of observation sequences
% Gamma  probability of current state cond. on data
% Xi     probability of past and future state cond. on data
%
% OUTPUT
% hmm    single hmm data structure with updated state model probs.
%
% Author: Diego Vidaurre, OHBA, University of Oxford


N = length(T);
K=hmm.K;
% transition matrix
sxi=squeeze(sum(Xi,1));   % counts over time
cutoff = hmm.train.cutoff; scutoff = sum(abs(cutoff));

hmm.Dir2d_alpha=sxi+hmm.prior.Dir2d_alpha;
PsiSum=psi(sum(hmm.Dir2d_alpha(:),1));
for j=1:K,
    for i=1:K,
        hmm.P(j,i)=exp(psi(hmm.Dir2d_alpha(j,i))-PsiSum);
    end;
    hmm.P(j,:)=hmm.P(j,:)./sum(hmm.P(j,:));
end;

% initial state
hmm.Dir_alpha=hmm.prior.Dir_alpha;
for in=1:N
    t = sum(T(1:in-1)) - (in-1)*scutoff + 1;
    hmm.Dir_alpha=hmm.Dir_alpha+Gamma(t,:);
end
PsiSum=psi(sum(hmm.Dir_alpha,2));
for i=1:K,
    hmm.Pi(i)=exp(psi(hmm.Dir_alpha(i))-PsiSum);
end
hmm.Pi=hmm.Pi./sum(hmm.Pi);

