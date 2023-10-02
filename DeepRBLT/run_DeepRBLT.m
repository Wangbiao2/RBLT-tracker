function results = run_DeepRBLT(seq, res_path, bSaveImage)

% Feature specific parameters
hog_params.cell_size = 4;
hog_params.compressed_dim = 13;  % 31---pca----->13


cnn_params.nn_name = 'imagenet-vgg-m-2048.mat'; % Name of the network
cnn_params.output_layer = [3 7 14];             % Which layers to use
cnn_params.downsample_factor = [2 1 1];         % How much to downsample each output layer
cnn_params.compressed_dim = [12 32 64];         % Compressed dimensionality of each output layer [96,256,512]
cnn_params.input_size_mode = 'adaptive';        % How to choose the sample size
cnn_params.input_size_scale = 1;                % Extra scale factor of the input samples to the network (1 is no scaling)

% Which features to include
params.t_features = {
    struct('getFeature',@get_cnn_layers, 'fparams',cnn_params),...
    struct('getFeature',@get_fhog,'fparams',hog_params),...
};

% Global feature parameters1s
params.t_global.normalize_power = 2;    % Lp normalization with this p
params.t_global.normalize_size = true;  % Also normalize with respect to the spatial size of the feature
params.t_global.normalize_dim = true;   % Also normalize with respect to the dimensionality of the feature
params.t_global.normalize_size = true;  % Also normalize with respect to the spatial size of the feature


% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 4.5;         % The scaling of the target size to get the search area
params.min_image_sample_size = 200^2;   % Minimum area of image samples
params.max_image_sample_size = 250^2;   % Maximum area of image samples

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 20;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

params.output_sigma_factor = 1/16;		% Label function sigma

% Interpolation parameters
params.interpolation_method = 'bicubic';    % The kind of interpolation kernel
params.interpolation_bicubic_a = -0.75;     % The parameter for the bicubic interpolation kernel
params.interpolation_centering = true;      % Center the kernel at the feature sample
params.interpolation_windowing = false;     % Do additional windowing on the Fourier coefficients of the kernel


params.reg_window_min = 1e-3; % The minimum value of the regularization window  's'
params.reg_window_max = 1e5;  % The maximum value of the regularization window  's'
params.background = 1.2;      % The maximum value of thr background-aware window  'w'
params.target = 0;			  % The minimum value of thr background-aware window  'w'


% Scale parameters for the translation model
% Only used if: params.use_scale_filter = false
params.number_of_scales = 7;            % Number of scales to run the detector
params.scale_step = 1.01;               % The scale factor
params.scale_step_refine = 1.01;        % re-fine scale factor, you can try 1.007,1.01

% params.number_of_scales = 5;            % Number of scales to run the detector
% params.scale_step = 1.012;               % The scale factor

% Scale filter parameters
% Only used if: params.use_scale_filter = true
params.use_scale_filter = false;         % Use the fDSST scale filter or not (for speed)

% GPU
params.use_gpu = true;                  % Enable GPU or not;  we use gpu in the DeepRBLT
params.gpu_id = [];                     % Set the GPU id, or leave empty to use default

% Initialize
params.seq = seq;

% TB-BiCF parameters
params.learning_rate{1,1,1} = 0.03;  % DeepRBLT's learning rate
params.learning_rate{1,1,2} = 0.02;
params.learning_rate{1,1,3} = 0.01;
params.learning_rate{1,1,4} = 0.032;

params.t_lambda = 1;                 % \gamma_{3} in the paper 

params.gamma1{1,1,1} = 13;           % \gamma_{1} in the paper
params.gamma1{1,1,2} = 7;
params.gamma1{1,1,3} = 3.5;
params.gamma1{1,1,4} = 17;

params.gamma2 = 0.055;              % \gamma_{2} in the paper

params.gamma3{1,1,1} = 10;          % \gamma_{4} in the paper 
params.gamma3{1,1,2} = 10;
params.gamma3{1,1,3} = 10;
params.gamma3{1,1,4} = 10;

% ADMM parameters
params.admm_iterations = 2;   % Iterations (for speed)  (3,100,500,100000) is also ok.
params.mu = 1500;             % Initial penalty factor
params.beta = 100;            % Scale step
params.mu_max = 100000;       % Maximum penalty factor

% Visualization
params.visualization = 1;     % Visualiza tracking
params.disp_fps = 1;

% Run tracker without coarse-to-fine scale search strategy
results = tracker_DeepRBLT(params);

% Run tracker with coarse-to-fine scale search strategy
% results = tracker_DeepRBLT_Scale(params);
