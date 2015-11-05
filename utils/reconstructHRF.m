function HRFs = reconstructHRF(hmm,prior)

if nargin<2, prior = 0; end
T = length(hmm.HRF);
if prior
    ndim = 1; 
else
    ndim = size(hmm.HRF(1).B.mu,2);
end
[p,L] = size(hmm.train.H);

HRFs = [];
for tr=1:T
    for n=1:ndim
        H = zeros(1,L);
        for j=1:p
            if prior
                H = H + hmm.HRF(tr).prior.B.mu(j,n) * hmm.train.H(j,:);
            else
                H = H + hmm.HRF(tr).B.mu(j,n) * hmm.train.H(j,:);
            end
        end
        HRFs = [HRFs; H];
    end
end

