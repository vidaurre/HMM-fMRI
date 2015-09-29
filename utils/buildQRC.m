function [Q,R,C] = buildQRC(hmm,tr)
% Auxiliar variables Q, R
ndim = size(hmm.HRF(tr).B.mu,2);
[p,L] = size(hmm.train.H);
Q = zeros(ndim,L); R = zeros(ndim,L); C = zeros(ndim,L-1,L); 
BB = zeros(p,p,ndim);
for n=1:ndim,
    BB(:,:,n) = hmm.HRF(tr).B.mu(:,n) * hmm.HRF(tr).B.mu(:,n)' + hmm.HRF(tr).B.S(:,:,n);
end

for l=1:L,
    ll = setdiff(1:L,l);
    for n=1:ndim
        Q(n,l) = hmm.train.H(:,l)' * BB(:,:,n) * hmm.train.H(:,l);
	for il = 1:length(ll)
		C(n,il,l) = hmm.train.H(:,ll(il))' * hmm.HRF(tr).B.S(:,:,n) * hmm.train.H(:,l);  
	end
    end
    R(:,l) = sum(repmat(hmm.train.H(:,l),1,ndim) .* hmm.HRF(tr).B.mu   )';
end
