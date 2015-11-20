function [G,V] = buildGV(Tfmri,T,hmm,X,n,tr)
% auxiliar variables G and V
% T refers to the fMRI data

p = size(hmm.train.H,1); ndim = size(X.mu,2);
cutoff = hmm.train.cutoff;
t0fMRI = sum(Tfmri(1:tr-1)); 
t0 = sum(T(1:tr-1)) + cutoff(1); t1 = sum(T(1:tr)) + cutoff(2);
Ttr = T(tr)-sum(abs(cutoff));
ind_diag = find(eye(ndim)==1); ind_diag = ind_diag(:);
obtain_V = nargout>1;

G = zeros(Tfmri(tr),p); 
if obtain_V, 
    V = zeros(p,p);
    ind_n = n : ndim : ( (Ttr-1)*ndim+n );
    Sx = X.S{tr}(ind_n,ind_n); % T x T cov matrix for this channel and epoch
end

for t=t0fMRI+1:t0fMRI+Tfmri(tr)
    ind_t = hmm.train.I2(t,:); % the x_t that predict y_t 
    in_boundary = (ind_t>t0 & ind_t<=t1); % exclude the boundaries
    ind_L = find(in_boundary); % the corresponding lags
    ind_t(~in_boundary) = []; % trimming the boundary points
    mx = X.mu(ind_t,n)'; % 1xL
    G(t-t0fMRI,:) = sum(repmat(mx,p,1) .* hmm.train.H(:,ind_L),2)';
    if obtain_V
        ind_t_t0 = ind_t - sum(T(1:tr-1)) - cutoff(1); % ind_t refers to the absolute position in Sx
        V = V + hmm.train.H(:,ind_L) * Sx(ind_t_t0,ind_t_t0) *  hmm.train.H(:,ind_L)';
    end
end
        
end
