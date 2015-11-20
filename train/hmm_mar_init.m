function [Gamma,hmm,fehist] = hmm_mar_init(X,T,options) 
%
% Initialise the hidden Markov chain using HMM-MAR
% X is initialize as in obsinit but using cross-validation
%
% INPUT
% data      observations, a struct with X (time series) and C (classes, optional)
% options,  structure with the training options  
%
% OUTPUT
% Gamma     p(state given X)
%
% Author: Diego Vidaurre, University of Oxford

addpath(genpath(options.HMMMAR_path))

train = struct('K',options.K);
train.covtype = options.covtype;
train.order = 0;
train.tol = 1e-7;
train.cyc = 100;
train.zeromean = 0;
train.inittype = 'GMM';
train.initcyc = 100;
train.initrep = 3;
train.DirichletDiag = options.DirichletDiag;
train.verbose = 0;

fehist = Inf;
for it=1:options.initrep
    [hmm0,Gamma0,~, ~, ~, ~, fehist0] = hmmmar(X,T,train);
    if size(Gamma0,2)<options.K
        Gamma0 = [Gamma0 0.0001*rand(size(Gamma0,1),options.K-size(Gamma0,2))];
        Gamma0 = Gamma0 ./ repmat(sum(Gamma0,2),1,options.K);
    end
    if options.verbose,
        fprintf('Init run %d, Free Energy %f \n',it,fehist0(end));
    end
    if fehist0(end)<fehist(end),
        fehist = fehist0; Gamma = Gamma0; hmm = hmm0; s = it; 
    end
end
if options.verbose
    fprintf('%i-th was the best iteration with FE=%f \n',s,fehist(end))
end
    
rmpath(genpath(options.HMMMAR_path))

end