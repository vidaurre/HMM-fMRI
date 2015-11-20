function [D] = gauss_kl (mu_q,mu_p,sigma_q,sigma_p,logdet_q,logdet_p,isigma_p)
% computes the divergence between two k-dimensional Gaussian prob. densities
%
%                /
%      D(q||p) = | q(x)*log(q(x)/p(x)) dx
%               /

if nargin<4,
  error('Incorrect number of input arguments');
end;

if length(mu_q)~=length(mu_p),
  error('Distributions must have equal dimensions (Means dimension)');
end;
mu_q=mu_q(:);
mu_p=mu_p(:);

if size(sigma_q)~=size(sigma_p),
  error('Distributions must have equal dimensions (Covariance dimension)');
end;

K=size(sigma_q,1);

if nargin<5, logdet_q = logdet(sigma_q,'chol'); end
if nargin<6, logdet_p = logdet(sigma_p,'chol'); end
if nargin<7, isigma_p = inv(sigma_p); end

D=logdet_p - logdet_q -K+trace(isigma_p*sigma_q)+(mu_q-mu_p)'*isigma_p*(mu_q-mu_p);
D=D*0.5;
