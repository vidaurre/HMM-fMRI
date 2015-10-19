function [I1,I2,T] = initindexes(TfMRI,L,HzfMRI,HzSignal)
% I1 (TsignalxL) says with y_t are predicted by each x_t1 for this lag (0 means none)
% I2 (TfMRIxL) says which x_t1 are predicting each y_t2
T = round((TfMRI-1) * (HzSignal/HzfMRI)) + L;

I1 = zeros(sum(T),L);
I2 = zeros(sum(TfMRI),L);

spacing = HzSignal/HzfMRI;
reye = eye(L); reye = reye(L:-1:1,:);
ind_rdiag = find(reye==1); ind_rdiag = ind_rdiag(:); 

rL = L:-1:1;

for in=1:length(T)
    tini_fMRI = sum(TfMRI(1:in-1));
    tini_X = sum(T(1:in-1));
    for t = 1:TfMRI(in)
        i = tini_X + 1 + spacing*(t-1);
        I = I1(i:i+L-1,:);
        %if any(I(ind_rdiag)>0), keyboard; end
        I(ind_rdiag) = I(ind_rdiag) + (tini_fMRI+t);
        I1(i:i+L-1,:) = I;
        I2(tini_fMRI+t,:) = rL + i - 1;
    end
end

end
