function [G,V] = buildGV(T,hmm,X,n)
% auxiliar variables G and V
% T refers to the fMRI data
ind_diag = find(eye(size(X.mu,2))==1); ind_diag = ind_diag(:); 
p = size(hmm.train.H,1);
G = zeros(T,p); 
if nargout>1, V = zeros(p,p); end
for t=1:T
    mx = X.mu(hmm.train.I2(t,:),n)'; % 1xL
    if strcmp(hmm.train.covtype,'diag')
        sx = X.S(hmm.train.I2(t,:),n)'; % 1xL
    else
        sx = X.S(hmm.train.I2(t,:),ind_diag(n))'; % 1xL
    end
    G(t,:) = sum(repmat(mx,p,1) .* hmm.train.H,2)';
    if nargout>1, V = V + (hmm.train.H .* repmat(sx,p,1)) *  hmm.train.H'; end
end
end