function [H,meanH,meanB,covB,Hz,cutoff] = HRFbasis(p,Hz,cutoffThres)
% FLOBS_hrfsamps.txt generated with Make_flobs_gui, 3 basis functions
% p is the number of basis functions (coefficients to estimate)
% Hz (input or output) refers to the frequency desired for the latent signal;
% cutoff_thresh
% otherwise it is as in Woolrich et al, 2004.
%
% Note: we cannot just reabsorb mu_M back into M and regress on H, 
% because H can't cope with this, it's not tuned for this, there'll be an error
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<3, cutoffThres = 0.90; end
if nargin<1, p = 3; end
M0 = dlmread('data/FLOBS_hrfsamps.txt'); % time points (L) by n samples
Hz0 = size(M0,1) / 28; % 28s is the time length - see FLOBS GUI
M = zeros(size(M0,1) * (round(Hz*1e2)/round(Hz0*1e2)),size(M0,2));
if nargin<2
    Hz = Hz0; % Hz0 is the freq of the built on HRF basis
else
    if Hz~=Hz0
        for j=1:size(M0,2)
            M(:,j) = resample(M0(:,j),round(Hz*1e2),round(Hz0*1e2));
        end
    end
end
L = size(M,1);
mu_M = mean(M); M = M - repmat(mu_M,L,1); %mu_M = mean(mu_M); % demeaning (needed for PCA)
[~,H] = pca(M,'NumComponents',p,'Centered',false);
H = [ ones(L,1) (H ./ repmat(std(H),L,1))]; % the intercept is necessary
M = M + repmat(mu_M,L,1); % put it back so that we don't have to carry it all the way through
meanH = mean(M,2)'; % 1 x L, 
R = pinv(H) * M; % p x L
meanB = mean(R,2); % you rescaled H, so you can't use the first argument of PCA, 
covB = cov(R');
H = H'; % p x L
cutoff = zeros(1,2);
acmeanH = cumsum(abs(meanH(end:-1:1))) / sum(abs(meanH)); 
cutoff(1) = find(acmeanH>=cutoffThres,1) - 1 ;
acmeanH = cumsum(abs(meanH)) / sum(abs(meanH));
cutoff(2) = 1-find(acmeanH>=cutoffThres,1,'first');

% H = ones(p,1);
% meanH = 0;

end



