function projection_matrix = update_projection_matrix(params,num_feature_blocks,yf,projection_matrix,xlf,xlf_proj,reg_energy,feature_dim,reg_filter,filter_sz,compressed_dim)
% Conjugate Gradient parameters
params.CG_iter = 5;                     % The number of Conjugate Gradient iterations in each update after the first frame
params.init_CG_iter = 10*15;            % The total number of Conjugate Gradient iterations used in the first frame
params.init_GN_iter = 10;               % The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated)
params.CG_use_FR = false;               % Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
params.CG_standard_alpha = true;        % Use the standard formula for computing the step length in Conjugate Gradient
params.CG_forgetting_rate = 50;	 	 	% Forgetting rate of the last conjugate direction
params.precond_data_param = 0.75;       % Weight of the data term in the preconditioner
params.precond_reg_param = 0.25;	 	% Weight of the regularization term in the preconditioner
params.precond_proj_param = 40;	 	 	% Weight of the projection matrix part in the preconditioner

% Factorized convolution parameters

params.proj_init_method = 'pca';        % Method for initializing the projection matrix
params.projection_reg = 1e-7;	 	 	% Regularization paremeter of the projection matrix

% Number of CG iterations per GN iteration 
init_CG_opts.maxit = ceil(params.init_CG_iter / params.init_GN_iter);

hf = cell(2,1,num_feature_blocks);
proj_energy = cellfun(@(P, yf) 2*sum(abs(yf(:)).^2) / sum(feature_dim) * ones(size(P), 'like', params.data_type), projection_matrix, yf, 'uniformoutput', false);
sample_energy = cellfun(@(xlf) abs(xlf .* conj(xlf)), xlf_proj, 'uniformoutput', false);

for k = 1:num_feature_blocks
    hf{1,1,k} = zeros([filter_sz(k,1) filter_sz(k,2) compressed_dim(k)], 'like', params.data_type_complex);
end


[~, projection_matrix, ~] = train_joint(hf, projection_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, params, init_CG_opts);