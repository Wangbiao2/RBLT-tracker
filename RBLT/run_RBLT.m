function results = run_RBLT(seq, rp, bSaveImage)
setup_paths();
% Feature specific parameters
hog_params.cell_size = 4;          % 31-dim hog 
hog_params.nDim = 31;

grayscale_params.colorspace='gray'; % 1-dim gray-scale
grayscale_params.cell_size = 4;

cn_params.tablename = 'CNnorm'; % 10-dim cn
cn_params.useForGray = false;  
cn_params.cell_size = 4;
cn_params.nDim = 10;

% Which features to include
params.t_features = {
    struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...    
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
};

% Global feature parameters1s
params.t_global.cell_size = 4;          % Feature cell size

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 5;           % The scaling of the target size to get the search area
params.min_image_sample_size = 150^2;   % Minimum area of image samples
params.max_image_sample_size = 200^2;   % Maximum area of image samples

% Spatial regularization window_parameters
params.feature_downsample_ratio = 4; %  Feature downsample ratio

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

% Learning parameters
params.output_sigma_factor = 1/16;		% Label function sigma

% scale filter (no cfse in the paper)
params.num_scales = 33;                 %from dsst
params.hog_scale_cell_size = 4;
params.learning_rate_scale = 0.025;
params.scale_sigma_factor = 1/2;
params.scale_model_factor = 1.0;
params.scale_step = 1.03;
params.scale_model_max_area = 32*16;
params.scale_lambda = 1e-4;

params.admm_iterations = 3;  

% Visualization
params.visualization = 1;              

params.learning_rate = 0.032;          % learning rate

params.reg_window_max = 1e5;           % The maximum value of the regularization window
params.reg_window_min = 1e-3;          % The minimum value of the regularization window
params.background = 1.2;               % The maximum value of the background-aware window 
params.target = 0.1;				   % The minimum value of the background-aware window 

params.gamma1 = 17;                    % \gamma_{1}(paper)
params.gamma2 = 0.055;                 % \gamma_{2}(paper)
params.gamma3 = 6;                     % \gamma_{4}(paper)  \gamma_{3}=1

% GPU
params.use_gpu = false;                 % Enable GPU or not, we do not use cpu in RBLT
params.gpu_id = [];                     % Set the GPU id, or leave empty to use default

% Initialize
params.seq = seq;

% Run tracker
results = tracker_RBLT(params);
