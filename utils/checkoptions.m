function options = checkoptions (options,data,cv)


if ~isfield(options,'K'), error('K was not specified'); end
if ~isfield(options,'Hz'), options.Hz = data.Hz; end
if ~isfield(options,'p'), options.p = 3; end % number of basis functions
if ~isfield(options,'covtype'), options.covtype = 'full'; end
if ~isfield(options,'zeroBmean'), options.zeroBmean = 0; end % prior mean of B
if ~isfield(options,'factorX'), options.factorX = 0; end % should we factorize the optimization of X over t
if ~isfield(options,'cyc'), options.cyc = 1000; end
if ~isfield(options,'tol'), options.tol = 1e-5; end
if ~isfield(options,'meancycstop'), options.meancycstop = 1; end
if ~isfield(options,'subcyc'), options.subcyc=[1 1]; end 
if ~isfield(options,'subcycHRF'), options.subcycHRF=[Inf 0]; end % until subcycHRF(1) it will estimate the HRF at all iterations,
% then it estimates it only when the iteration number is divisible by subcycHRF(2)
if ~isfield(options,'stopwhenK1'), options.stopwhenK1=0; end % should we stop training if only 1 state remains?
if ~isfield(options,'cycstogoafterevent'), options.cycstogoafterevent = 20; end
if ~isfield(options,'initcyc'), options.initcyc = 100; end
if ~isfield(options,'initrep'), options.initrep = 5; end
if ~isfield(options,'inittype'), options.inittype = 'GMM'; end
if ~isfield(options,'DirichletDiag'), options.DirichletDiag = 2; end
if ~isfield(options,'cutoffThres'), options.cutoffThres = 0.95; end % Initial and final part of each trial of 
if ~isfield(options,'beta'), options.beta = 1e-2; end % scaling factor on the prior covariance for the HRF parameters
% X will not be considered to compute the states parameters and will not be
% updated either - this is used to define this. 
if ~isfield(options,'lambda'), options.lambda = 0; end % regularisation parameter in the obs init
if ~isfield(options,'Gamma'), options.Gamma = []; end
if ~isfield(options,'hmm'), options.hmm = []; end
if ~isfield(options,'X'), options.X = []; end
if ~isfield(options,'repetitions'), options.repetitions = 1; end
if ~isfield(options,'updateGamma'), options.updateGamma = 1; end
if ~isfield(options,'verbose'), options.verbose = 1; end


if mod(options.Hz,data.Hz)>0, error('options.Hz has to be a multiple of data.Hz'); end
options.HzfMRI = data.Hz;

if ~strcmp(options.inittype,'random') && options.initrep == 0,
    options.inittype = 'random';
    warning('GMM init was set, but initrep==0 - setting back to random..')
end

if options.K>1 && options.updateGamma == 0 && isempty(options.Gamma), 
    warning('Gamma is unspecified, so updateGamma was set to 1');  options.updateGamma = 1; 
end
if options.updateGamma == 0 && options.repetitions>1,
    error('If Gamma is not going to be updated, repetitions>1 is unnecessary')
end


if cv==1
    if ~isfield(options,'cvfolds'), options.cvfolds = length(data.T); end
    if ~isfield(options,'cvrep'), options.cvrep = 1; end
    if ~isfield(options,'cvmode'), options.cvmode = 1; end
    if ~isfield(options,'cvverbose'), options.cvverbose = 0; end
    if length(options.cvfolds)>1 && length(options.cvfolds)~=length(data.T), error('Incorrect assigment of trials to folds'); end
    if length(options.cvfolds)>1 && ~isempty(options.Gamma), error('Set options.Gamma=[] for cross-validating'); end
    if length(options.cvfolds)==1 && options.cvfolds==0, error('Set options.cvfolds to a positive integer'); end
    if options.K==1 && isfield(options,'cvrep')>1, warning('If K==1, cvrep>1 has no point; cvrep is set to 1 \n'); end
    if ~isempty(options.checkpt_fname), warning('Intermediate files are not saved for CV \n'); options.checkpt_fname = ''; end
end

end



