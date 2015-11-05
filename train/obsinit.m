function [X,updateXindexes] = obsinit(data,hmm)
% updateXindexes refers to indexes of the time points in X that are
% going to be actualized during VB training. The time points at the
% beginning and the end of each trial are only estimated by obsinit and
% they are left intact for the rest of the process

meanH = hmm.train.meanH; 
L = length(meanH);
rmeanH = meanH(L:-1:1); 
T = data.T + L - 1;
ndim = size(data.Y,2);
cutoff = hmm.train.cutoff; scutoff = sum(abs(cutoff));
updateXindexes = [];

X.mu = zeros(sum(T),ndim);
if hmm.train.factorX
    if strcmp(hmm.train.covtype,'diag')
        X.S = zeros(sum(T),ndim);
    else
        X.S = zeros(sum(T),ndim,ndim);
    end
else
    X.S = cell(length(T),1);
end

for tr=1:length(data.T)
    t0fMRI = sum(data.T(1:tr-1)); t1fMRI = sum(data.T(1:tr));
    t0 = sum(T(1:tr-1)); t1 = sum(T(1:tr));
    updateXindexes = [updateXindexes (t0+cutoff(1)+1):(t1+cutoff(2))];
    Ttr = T(tr)-scutoff; 
    if tr==1 || data.T(tr)~=data.T(tr-1)
        A = zeros(data.T(tr),T(tr)); % estimation done for each channel independently
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
    if ~hmm.train.factorX, X.S{tr} = zeros(Ttr*ndim); end
    for n=1:ndim
        b = data.Y(t0fMRI+1:t1fMRI,n);
        %X.mu(t0+1:t1,n) = pinv(A)*b;
        if hmm.train.factorX
            if strcmp(hmm.train.covtype,'diag')
                X.S(t0+1:t1,n) = diag(S);
            else
                X.S(t0+1:t1,n,n) = diag(S);
            end
        else
            ind1 = n : ndim : ( (Ttr-1)*ndim+n );
            ind2 = (cutoff(1)+1):(cutoff(1)+Ttr);
            X.S{tr}(ind1,ind1) = S(ind2,ind2);
        end
        X.mu(t0+1:t1,n) = S * A' * b;
        %X.mu(t0+1:t1,n) = X.mu(t0+1:t1,n) / std(X.mu(t0+1:t1,n));
    end
end

end
