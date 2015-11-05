function [hmm, Gamma, Xi, vpath, X, fehist] = hmmfmri (data,options)
%
% Main function to train the HMM-MAR model, compute the Viterbi path and,
% if requested, obtain the cross-validated sum of prediction quadratic errors. 
% 
% INPUT
% data          observations, a struct with Y (time series), T (length of fmri series),
%               Hz (sampling rate) and C (classes, optional) 
% options       structure with the training options - see documentation
%
% OUTPUT
% hmm           estimated HMMMAR model 
% Gamma         Time courses of the states probabilities given data
% Xi            joint probability of past and future states conditioned on data
% vpath         most likely state path of hard assignments
% X             latent signal
% GammaInit     Time courses used after initialization.
% fehist        historic of the free energies across iterations
%
% Author: Diego Vidaurre, OHBA, University of Oxford

options = checkoptions(options,data,0);
[options.H,options.meanH,options.meanB,options.covB,~,options.cutoff] = ...
    HRFbasis(options.p,options.Hz,options.cutoffThres);
[options.I1,options.I2,T] = initindexes(data.T,size(options.H,2),data.Hz,options.Hz);

if ~isfield(data,'C'), 
    if options.K>1, data.C = NaN(sum(T),options.K); 
    else data.C = ones(sum(T),1); 
    end
elseif options.K~=size(data.C,2), 
    error('Matrix data.C should have K columns');
end

if isempty(options.hmm) % Initialisation of the hmm
    hmm_wr = struct('train',struct());
    hmm_wr.K = options.K;
    hmm_wr.train = options; 
    %if options.whitening, hmm_wr.train.A = A; hmm_wr.train.iA = iA;  end
    hmm_wr=hmmhsinit(hmm_wr);
    [hmm_wr,X_wr,Gamma_wr]=hmminit(data,T,hmm_wr);
else % using a warm restart from a previous run
    hmm_wr = options.hmm;
    options = rmfield(options,'hmm');
    hmm_wr.train = options; 
    Gamma_wr = options.Gamma;
    X_wr = options.X;
end

fehist = Inf;
for it=1:options.repetitions
    hmm0 = hmm_wr; Gamma0 = Gamma_wr; X0 = X_wr;
    [hmm0,Gamma0,Xi0,X0,fehist0] = hmmtrain(data,T,hmm0,X0,Gamma0);
    if options.updateGamma==1 && fehist0(end)<fehist(end),
        fehist = fehist0; hmm = hmm0; 
        Gamma = Gamma0; Xi = Xi0; X = X0;
    elseif options.updateGamma==0,
        fehist = []; hmm = hmm0; 
        Gamma = options.Gamma; Xi = []; X = X0;
    end
end

if options.updateGamma
    vp = hmmdecode(T,hmm,X);
    vpath=[];
    for in=1:length(vp)
        vpath = [vpath; vp(in).q_star];
    end
else
    vpath = ones(size(Gamma,1),1);
end
 
end