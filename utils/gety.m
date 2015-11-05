function y = gety(data,T,tr,hmm,X)
% returns fmri data to be used in the estimation of B and X,
% where the the start and the end of the trial need to be adjusted, 
% as it does not participate in the estimation (excluded to avoid boundary effects) 

ndim = size(X.mu,2);
cutoff = hmm.train.cutoff;
t0fMRI = sum(data.T(1:tr-1)); t1fMRI = sum(data.T(1:tr));
y = data.Y(t0fMRI+1:t1fMRI,:);
meanH = hmm.train.meanH; 
r = [(sum(T(1:tr-1))+1) : (sum(T(1:tr-1))+cutoff(1)) ... % first time points of the trial
    (sum(T(1:tr))+cutoff(2)+1) : sum(T(1:tr))]; % last time points of the trial
for t = r  
    these_l = hmm.train.I1(t,:) > 0;
    these_y = hmm.train.I1(t,these_l);
    for n=1:ndim
        y(these_y-t0fMRI,n) = y(these_y-t0fMRI,n) - (X.mu(t,n) * meanH(these_l))';
    end
end
end