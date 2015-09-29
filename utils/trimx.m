function Y = trimx (X,T,cutoff)

Y = struct('mu',[],'S',[]);
for tr=1:length(T)
    t0 = sum(T(1:tr-1)) + cutoff(1); t1 = sum(T(1:tr)) + cutoff(2);
    Y.mu = cat(1,Y.mu,X.mu(t0+1:t1,:));
    Y.S = cat(1,Y.S,X.S(t0+1:t1,:));
end
