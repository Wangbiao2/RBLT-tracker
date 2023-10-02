function wf = train_DeepTriCF(params, mf, yf, s,mf_p, wf_p,xf,xf_p,zf)

Smy = cellfun(@(mf, yf) bsxfun(@times, mf, conj(yf)), mf, yf, 'UniformOutput', false);
Smm = cellfun(@(mf) bsxfun(@times, mf, conj(mf)), mf, 'UniformOutput', false);
Smmp = cellfun(@(mf,mf_p) bsxfun(@times, mf, conj(mf_p)), mf,mf_p, 'UniformOutput', false);
x_fuse = cellfun(@(xf, xf_p) xf + xf_p, xf, xf_p, 'UniformOutput', false);
Sxx_fuse = cellfun(@(x_fuse) bsxfun(@times, x_fuse, conj(x_fuse)), x_fuse, 'UniformOutput', false);
Szz = cellfun(@(zf) bsxfun(@times, zf, conj(zf)), zf, 'UniformOutput', false);
ss = cellfun(@(s) bsxfun(@times, s, conj(s)),s, 'UniformOutput', false);

% feature size
sz = cellfun(@(xf) size(xf), xf, 'UniformOutput', false);
N = cellfun(@(sz) sz(1) * sz(2), sz, 'UniformOutput', false);

% initialize hs
hf = cellfun(@(sz) zeros(sz), sz, 'UniformOutput', false);

% initialize lagrangian multiplier
zetaf = cellfun(@(sz) zeros(sz), sz, 'UniformOutput', false);

% penalty
mu = params.mu;
beta = params.beta;
mu_max = params.mu_max;
lambda = params.t_lambda;
gamma1 = params.gamma1;
gamma2 = params.gamma2;
gamma3 = params.gamma3;

iter = 1;
while (iter <= params.admm_iterations)
    wf = cellfun(@(Smy, wf_p, Smm,Smmp, Sxx_fuse,Szz, hf, zetaf,gamma1) ...
        bsxfun(@rdivide, ...
		Smy + bsxfun(@times, wf_p, gamma1*Smmp+gamma2*Sxx_fuse)+ mu * hf - zetaf, ...
		(1+gamma1)*Smm + gamma2 * Sxx_fuse+ gamma3 * Szz + mu), ...
		Smy, wf_p, Smm,Smmp, Sxx_fuse,Szz, hf, zetaf,gamma1, 'UniformOutput', false);
    
    if iter == params.admm_iterations
        break;
    end
    hf = cellfun(@(wf, zetaf, N,ss) fft2(bsxfun(@rdivide, (ifft2(mu * wf + zetaf)), lambda/N * ss + mu)), wf, zetaf, N,ss, 'UniformOutput', false);
    zetaf = cellfun(@(zetaf, wf, hf) zetaf + mu * wf - mu * hf, zetaf, wf, hf, 'UniformOutput', false);
    mu = min(mu_max, beta * mu);
    iter = iter + 1;
end
