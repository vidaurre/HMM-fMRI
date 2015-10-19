function [X,updateXindexes] = obsinit(data,hmm)

meanH = hmm.train.meanH; 
L = length(meanH);
rmeanH = meanH(L:-1:1); 
T = data.T + L - 1;
ndim = size(data.Y,2);
cutoff = hmm.train.cutoff;
updateXindexes = [];

X.mu = zeros(sum(T),ndim);
if strcmp(hmm.train.covtype,'diag')
    X.S = zeros(sum(T),ndim);
else
    X.S = zeros(sum(T),ndim,ndim); 
end

for tr=1:length(data.T)
    t0fMRI = sum(data.T(1:tr-1)); t1fMRI = sum(data.T(1:tr));
    t0 = sum(T(1:tr-1)); t1 = sum(T(1:tr));
    updateXindexes = [updateXindexes (t0+cutoff(1)+1):(t1+cutoff(2))];
    if tr==1 || data.T(tr)~=data.T(tr-1)
        A = zeros(data.T(tr),T(tr));
        for t=1:data.T(tr)
            ind = (1:L) + (t-1);
            A(t,ind) = rmeanH;
        end
        C = A' * A;
        if hmm.train.lambda==0
            lambda = 0.1 * mean(diag(C));
        else
            lambda = hmm.train.lambda;
        end
        S = inv(C + lambda * eye(size(A,2)));
    end
    for n=1:ndim
        b = data.Y(t0fMRI+1:t1fMRI,n);
        %X.mu(t0+1:t1,n) = pinv(A)*b;
        if strcmp(hmm.train.covtype,'diag')
            X.S(t0+1:t1,n) = diag(S);
        else
            X.S(t0+1:t1,n,n) = diag(S);
        end
        X.mu(t0+1:t1,n) = S * A' * b;
        %X.mu(t0+1:t1,n) = X.mu(t0+1:t1,n) / std(X.mu(t0+1:t1,n)); 
    end
end

end
