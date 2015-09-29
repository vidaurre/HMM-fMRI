function [H,meanH,meanB,covB,Hz,cutoff] = HRFbasis(p,Hz,zeroBmean,cutoffThres)
% FLOBS_hrfsamps.txt generated with Make_flobs_gui, 3 basis functions
% p is the number of basis functions (coefficients to estimate)
% Hz (input or output) refers to the frequency desired for the latent signal;
% zeroBmean=1 tells that the mean of the distribution of B is 0; 
% cutoff_thresh
% otherwise it is as in Woolrich et al, 2004.
%
% Author: Diego Vidaurre, OHBA, University of Oxford


if nargin<3, zeroBmean = 1; end
if nargin<4, cutoffThres = 0.90; end
if nargin<1, p = 3; end
M = dlmread('data/FLOBS_hrfsamps.txt'); % time points (L) by n samples
L = size(M,1);
Hz0 = L / 28; % 28s is the time length - see FLOBS GUI
[~,H] = pca(M,'NumComponents',p);
H = H ./ repmat(std(H),L,1);
meanH = mean(M,2)'; % 1 x L
R = pinv(H) * M; % p x L
if zeroBmean==1
    meanB = zeros(p,1);
else
    meanB = mean(R,2);
end
covB = cov(R');
if nargin<2
    Hz = Hz0; % Hz0 is the freq of the built on HRF basis
else
    if Hz~=Hz0
        H0 = H; H = zeros(round(L*Hz/Hz0),p);
        for j=1:p
            H(:,j) = resample(H0(:,j),round(Hz*1e2),round(Hz0*1e2));
        end
        meanH = resample(meanH,round(Hz*1e2),round(Hz0*1e2)); 
    end
end
H = H'; % p x L
cutoff = zeros(1,2);
acmeanH = cumsum(abs(meanH(end:-1:1))) / sum(abs(meanH)); 
cutoff(1) = find(acmeanH>=cutoffThres,1) - 1 ;
acmeanH = cumsum(abs(meanH)) / sum(abs(meanH));
cutoff(2) = 1-find(acmeanH>=cutoffThres,1,'first');

% H = ones(p,1);
% meanH = 0;

end



