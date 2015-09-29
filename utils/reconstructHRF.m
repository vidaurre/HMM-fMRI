function HRFs = reconstructHRF(hmm)

T = length(hmm.HRF);
ndim = size(hmm.HRF(1).B.mu,2);
[p,L] = size(hmm.train.H);

HRFs = [];
for tr=1:T
    for n=1:ndim
        H = zeros(1,L);
        for j=1:p
            H = H + hmm.HRF(tr).B.mu(j,n) * hmm.train.H(j,:);
        end
        HRFs = [HRFs; H];
    end
end

