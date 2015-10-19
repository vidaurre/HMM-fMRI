function [Q,R] = buildQR(hmm,tr)
% Auxiliar variables Q, R for a non-factorized X
ndim = size(hmm.HRF(tr).B.mu,2);
[p,L] = size(hmm.train.H);
Q = zeros(L,L,ndim); R = zeros(ndim,L); 
%BB = zeros(p,p,ndim);
%for n=1:ndim,
%    BB(:,:,n) = hmm.HRF(tr).B.mu(:,n) * hmm.HRF(tr).B.mu(:,n)' + hmm.HRF(tr).B.S(:,:,n);
%end
for l=1:L,
    R(:,l) = sum(repmat(hmm.train.H(:,l),1,ndim) .* hmm.HRF(tr).B.mu   )';
end
for l1=1:L,
    for l2=1:L
        for n=1:ndim
            Q(l1,l2,n) = hmm.train.H(:,l)' * hmm.HRF(tr).B.S(:,:,n) * hmm.train.H(:,l2);
        end
    end
end
    
