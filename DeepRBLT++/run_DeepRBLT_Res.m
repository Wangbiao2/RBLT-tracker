function results = run_DeepRBLT_Res(seq, res_path, bSaveImage)
setup_paths();
%% Set feature parameters
% HOG feature settings
hog_params.cell_size = 4;           
hog_params.feature_is_deep = false;

   
% CN feature settings
cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.feature_is_deep = false;   


% CNN feature settings
dagnn_params.nn_name = 'imagenet-resnet-50-dag.mat'; 
dagnn_params.output_var = {'res4ex'};    
dagnn_params.feature_is_deep = true;
dagnn_params.augment.blur = 1;
dagnn_params.augment.rotation = 1;
dagnn_params.augment.flip = 1;

params.t_features = {
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_dagnn_layers, 'fparams',dagnn_params),...
};

% Set [non-deep deep] parameters
params.learning_rate{1,1,1} = 0.032;       
params.learning_rate{1,1,2} = 0.032;
params.learning_rate{1,1,3} = 0.007;

params.output_sigma_factor = [1/16 1/5];	% desired label setting

params.reg_window_min = 1e-3; 
params.reg_window_max = 1e5;  
params.background = 1.2;                       
params.target = 0;				

params.t_lambda = 1;
params.gamma1{1,1,1} = 17;
params.gamma1{1,1,2} = 17;
params.gamma1{1,1,3} = 3; 

params.gamma2 = 0.055;
params.gamma3 = 10;

params.admm_iterations = 2;   % Iterations
params.mu = 1500;              % Initial penalty factor
params.beta = 100;             % Scale step
params.mu_max = 100000;       % Maximum penalty factor

% Image sample parameters
params.search_area_scale = [4.5 4.5];        % search region  
params.min_image_sample_size = [200^2 200^2];% minimal search region size   
params.max_image_sample_size = [250^2 250^2];% maximal search region size  

% Detection parameters
params.refinement_iterations = 1;           % detection numbers
params.newton_iterations = 15;               % subgrid localisation numbers   

% Set scale parameters
params.number_of_scales = 7;                % scale pyramid size
params.scale_step = 1.01;                   % scale step ratio

% Set GPU 
params.use_gpu = true;                 
params.gpu_id = [];              

% Initialisation
params.vis_res = 0;                         % visualisation results
params.vis_details = 0;                     % visualisation details for debug
params.seq = seq;   

% Run tracker
[results] = tracker_DeepRBLT_Res(params);