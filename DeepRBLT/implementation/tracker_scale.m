function results = tracker_scale(params)
SEM_H = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get sequence info
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');
if isempty(im)
    seq.rect_position = [];
    [seq, results] = get_sequence_results(seq);
    return;
end

% Init position
pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;

% Feature settings
features = params.t_features;

% Set default parameters
params = init_default_params(params);

% Global feature parameters
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;

% Define data types
if params.use_gpu
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end

params.data_type_complex = complex(params.data_type);
global_fparams.data_type = params.data_type;

init_target_sz = target_sz;

% Check if color image
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        is_color_image = false;
    else
        is_color_image = true;
    end
else
    is_color_image = false;
end

if size(im,3) > 1 && is_color_image == false
    im = im(:,:,1);
end
if is_color_image   %  Feature downsample ratio (We found that decreasing the downsampling ratios of CNN layer may benefit the performance)
    params.feature_downsample_ratio = [4,8,14,4,4];
else
    params.feature_downsample_ratio = [4,8,14,4];
end

params.weight_downsample_ratio = [4,8,14,4,4];

% params.feature_downsample_ratio = [4,8, 14 ,4,4];

% Check if mexResize is available and show warning otherwise.
params.use_mexResize = true;
global_fparams.use_mexResize = true;
try
    [~] = mexResize(ones(5,5,3,'uint8'), [3 3], 'auto');
catch err
    warning('DeepDMCF:tracker', 'Error when using the mexResize function. Using Matlab''s interpolation function instead, which is slower.\nTry to run the compile script in "external_libs/mexResize/".\n\nThe error was:\n%s', getReport(err));
    params.use_mexResize = false;
    global_fparams.use_mexResize = false;
end

% Calculate search area and initial scale factor
search_area = prod(init_target_sz * params.search_area_scale);
if search_area > params.max_image_sample_size
    currentScaleFactor = sqrt(search_area / params.max_image_sample_size);
elseif search_area < params.min_image_sample_size
    currentScaleFactor = sqrt(search_area / params.min_image_sample_size);
else
    currentScaleFactor = 1.0;
end

currentScaleFactor_1 = currentScaleFactor;
currentScaleFactor_2 = currentScaleFactor;

% target size at the initial scale
% base_target_sz = target_sz / currentScaleFactor;
base_target_sz = target_sz ./ [currentScaleFactor_1,currentScaleFactor_2];
init_bt_sz = base_target_sz;
beg_sz = base_target_sz;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        img_sample_sz = floor( base_target_sz * params.search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        img_sample_sz = repmat(sqrt(prod(base_target_sz * params.search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        img_sample_sz = base_target_sz + sqrt(prod(base_target_sz * params.search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    case 'custom'
        img_sample_sz = [base_target_sz(1)*2 base_target_sz(2)*2]; % for testing
end

[features, global_fparams, feature_info] = init_features(features, global_fparams, is_color_image, img_sample_sz, 'odd_cells');

% set compressed_dim ;
if is_color_image
    compressed_dim = [
        params.t_features{1, 1}.fparams.compressed_dim(1);...
        params.t_features{1, 1}.fparams.compressed_dim(2);...
        params.t_features{1, 1}.fparams.compressed_dim(3);...
        params.t_features{1, 2}.fparams.compressed_dim;...
%         1;...
];
else
    compressed_dim = [
        params.t_features{1, 1}.fparams.compressed_dim(1);...
        params.t_features{1, 1}.fparams.compressed_dim(2);...
        params.t_features{1, 1}.fparams.compressed_dim(3);...
        params.t_features{1, 2}.fparams.compressed_dim;...
%     1
];
end

% Set feature info
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
feature_dim = feature_info.dim;
num_feature_blocks = length(feature_dim);
feature_cell_sz = unique(feature_info.min_cell_size, 'rows', 'stable');

% Get feature specific parameters
feature_extract_info = get_feature_extract_info(features);

% Size of the extracted feature maps
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% Number of Fourier coefficients to save for each filter layer. This will
% be an odd number.
% filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz = feature_sz ;
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

% The size of the label function DFT. Equal to the maximum filter size.
[output_sz, k1] = max(filter_sz, [], 1);
k1 = k1(1);

% Get the remaining block indices
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];

% How much each feature block has to be padded to the obtain output_sz
% pad_sz = cellfun(@(filter_sz) (output_sz - filter_sz) / 2, filter_sz_cell, 'uniformoutput', false);
% pad_sz = cellfun(@(pad_sz) cast(pad_sz, 'like', params.data_type), pad_sz, 'uniformoutput', false);
% Construct the Gaussian label function
yf = permute(cell(num_feature_blocks, 1), [2 3 1]);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma = sqrt(prod(floor(base_target_sz/feature_cell_sz))) * params.output_sigma_factor;
    rg           = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg           = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]     = ndgrid(rg,cg);
    y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
    yf{1, 1, i}  = fft2(y);
end
yf = cellfun(@(yf) cast(yf, 'like', params.data_type), yf, 'uniformoutput', false);


% construct cosine window
cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);


% reg_window 
reg_window = cell(1,1,num_feature_blocks);
weight_window = cell(1,1,num_feature_blocks);
for i = 1:num_feature_blocks
    reg_scale = floor(base_target_sz/params.feature_downsample_ratio(i));
    use_sz = filter_sz_cell{1,1,i};    
    reg_window{1,1,i} = ones(use_sz) * params.reg_window_max;
    range = zeros(numel(reg_scale), 2);
    
    % determine the target center and range in the regularization windows
    for j = 1:numel(reg_scale)
        range(j,:) = [0, reg_scale(j) - 1] - floor(reg_scale(j) / 2);
    end
    center = floor((use_sz + 1)/ 2) + mod(use_sz + 1,2);
    range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
    range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));
    
    reg_window{1,1,i}(range_h, range_w) = params.reg_window_min;
end

for ii = 1:num_feature_blocks
    weight_scale = floor(base_target_sz/params.weight_downsample_ratio(ii));
    weight_sz = filter_sz_cell{1,1,ii};    
    weight_window{1,1,ii} = ones(weight_sz) * params.sur;
    range_weight = zeros(numel(weight_scale), 2);
    
    % determine the target center and range in the regularization windows
    for jj = 1:numel(weight_scale)
        range_weight(jj,:) = [0, weight_scale(j) - 1] - floor(weight_scale(j) / 2);
    end
    center_weight = floor((weight_sz + 1)/ 2) + mod(weight_sz + 1,2);
    range_h_weight = (center_weight(1)+ range_weight(1,1)) : (center_weight(1) + range_weight(1,2));
    range_w_weight = (center_weight(2)+ range_weight(2,1)) : (center_weight(2) + range_weight(2,2));
    
    weight_window{1,1,ii}(range_h_weight,range_w_weight) = params.center;
end


reg_window = cellfun(@(reg_window) cast(reg_window, 'like', params.data_type), reg_window, 'uniformoutput', false);
weight_window = cellfun(@(weight_window) cast(weight_window, 'like', params.data_type), weight_window, 'uniformoutput', false);


% Pre-computes the grid that is used for socre optimization
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;

% if params.use_scale_filter
%     [nScales, scale_step, scaleFactors, scale_filter, params] = init_scale_filter(params);
% else
    % Use the translation filter to estimate the scale.
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;
% end

if nScales > 0
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ img_support_sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

seq.time = 0;

% Allocate
scores_fs_feat = cell(1,1,num_feature_blocks);

while true
    % Read image
    if seq.frame > 0
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        if size(im,3) > 1 && is_color_image == false
            im = im(:,:,1);
        end
    else
        seq.frame = 1;
    end
    
    frame_tic = tic();
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Target localization step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Do not estimate translation and scaling on the first frame, since we
    % just want to initialize the tracker there
    if seq.frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            sample_pos = round(pos);
            det_sample_pos = sample_pos;
            sample_scale_1 = currentScaleFactor_1*scaleFactors;
            sample_scale_2 = currentScaleFactor_2*scaleFactors;
            xt = extract_features_new(im, sample_pos, sample_scale_1,sample_scale_2, features, global_fparams, feature_extract_info);
            
            xt_proj = project_sample(xt, projection_matrix);
            % Do windowing of features
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_proj, cos_window, 'uniformoutput', false);
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            % Do windowing of features
            scores_fs_feat{k1} = sum(bsxfun(@times, conj(wf{k1}), xtf{k1}), 3);
            scores_fs_sum = scores_fs_feat{k1};
            for k = block_inds
                scores_fs_feat{k} = sum(bsxfun(@times, conj(wf{k}), xtf{k}), 3);
                scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
            end
            % Also sum over all feature blocks.
            % Gives the fourier coefficients of the convolution response.
            scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]); % gpu->cpu
            response = ifft2(scores_fs, 'symmetric');
            
            [trans_row, trans_col, scale_ind] = resp_newton(response, scores_fs, newton_iterations, ky, kx, output_sz);
            
            if scale_ind>1 && scale_ind<7
                refineFactors_1 = [scaleFactors(scale_ind),scaleFactors(scale_ind),scaleFactors(scale_ind),scaleFactors(scale_ind-1),scaleFactors(scale_ind+1)];
                refineFactors_2 = [scaleFactors(scale_ind-1),scaleFactors(scale_ind+1),scaleFactors(scale_ind),scaleFactors(scale_ind),scaleFactors(scale_ind)];
                scale_refine_1 = currentScaleFactor_1*refineFactors_1;
                scale_refine_2 = currentScaleFactor_2*refineFactors_2;
                xt = extract_features_new(im, sample_pos, scale_refine_1,scale_refine_2, features, global_fparams, feature_extract_info);

                xt_proj = project_sample(xt, projection_matrix);
                % Do windowing of features
                xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_proj, cos_window, 'uniformoutput', false);
                xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
                % Do windowing of features
                scores_fs_feat{k1} = sum(bsxfun(@times, conj(wf{k1}), xtf{k1}), 3);
                scores_fs_sum = scores_fs_feat{k1};
                for k = block_inds
                    scores_fs_feat{k} = sum(bsxfun(@times, conj(wf{k}), xtf{k}), 3);
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                    scores_fs_sum = scores_fs_sum +  scores_fs_feat{k};
                end
                % Also sum over all feature blocks.
                % Gives the fourier coefficients of the convolution response.
                scores_fs = permute(gather(scores_fs_sum), [1 2 4 3]); % gpu->cpu
                response = ifft2(scores_fs, 'symmetric');
                [trans_row, trans_col, scale_refine_ind] = resp_newton(response, scores_fs, newton_iterations, ky, kx, output_sz);   
            end
            SEM = sem(response);
                if seq.frame > 10
                    if (SEM - mean(SEM_H))/std(SEM_H)>2
                        params.gamma2 = 0;
                    else
                        params.gamma2 = 0.055;
                    end
                end
            
            SEM_H = [SEM_H,SEM];

            % Compute the translation vector in pixel-coordinates and round
            % to the closest integer pixel.
            if scale_ind==1 || scale_ind==7
                translation_vec = [trans_row, trans_col] .* (img_support_sz./output_sz) .* [currentScaleFactor_1, currentScaleFactor_2] *scaleFactors(scale_ind);
                scale_change_factor = [scaleFactors(scale_ind),scaleFactors(scale_ind)];
            else
                translation_vec = [trans_row, trans_col] .* (img_support_sz./output_sz) .* [currentScaleFactor_1, currentScaleFactor_2] .*[refineFactors_1(scale_refine_ind),refineFactors_2(scale_refine_ind)];
                scale_change_factor = [refineFactors_1(scale_refine_ind),refineFactors_2(scale_refine_ind)];
            end
            % update position
            old_pos = pos;
            pos = sample_pos + translation_vec;
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
                    
            % Do scale tracking with the scale filter
%             if nScales > 0 && params.use_scale_filter
%                 scale_change_factor = scale_filter_track(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
%             end

            % Update the scale
            currentScaleFactor = [currentScaleFactor_1,currentScaleFactor_2].* scale_change_factor;
            currentScaleFactor_1 = currentScaleFactor(1);
            currentScaleFactor_2 = currentScaleFactor(2);
            % Adjust to make sure we are not to large or to small
            if currentScaleFactor_1 < min_scale_factor
                currentScaleFactor_1 = min_scale_factor;
            elseif currentScaleFactor_1 > max_scale_factor
                currentScaleFactor_1 = max_scale_factor;
            end  
            if currentScaleFactor_2 < min_scale_factor
                currentScaleFactor_2 = min_scale_factor;
            elseif currentScaleFactor_2 > max_scale_factor
                currentScaleFactor_2 = max_scale_factor;
            end
            iter = iter + 1;
        end
%             if currentScaleFactor_1~=currentScaleFactor_2
%                 Arc = currentScaleFactor_1/currentScaleFactor_2;
%                 beg_sz(1) = init_bt_sz(1)*sqrt(Arc);
%                 beg_sz(2) = init_bt_sz(2)/sqrt(Arc);

%                 for i = 1:num_feature_blocks
%                     reg_scale = floor(beg_sz/params.feature_downsample_ratio(i));
%                     use_sz = filter_sz_cell{1,1,i};    
%                     reg_window{1,1,i} = ones(use_sz) * params.reg_window_max;
%                     range = zeros(numel(reg_scale), 2);
% 
%                     % determine the target center and range in the regularization windows
%                     for j = 1:numel(reg_scale)
%                         range(j,:) = [0, reg_scale(j) - 1] - floor(reg_scale(j) / 2);
%                     end
%                     center = floor((use_sz + 1)/ 2) + mod(use_sz + 1,2);
%                     range_h = (center(1)+ range(1,1)) : (center(1) + range(1,2));
%                     range_w = (center(2)+ range(2,1)) : (center(2) + range(2,2));
% 
%                     reg_window{1,1,i}(range_h, range_w) = params.reg_window_min;
%                 end
% 
%                 for ii = 1:num_feature_blocks
%                     weight_scale = floor(beg_sz/params.weight_downsample_ratio(ii));
%                     weight_sz = filter_sz_cell{1,1,ii};    
%                     weight_window{1,1,ii} = ones(weight_sz) * params.sur;
%                     range_weight = zeros(numel(weight_scale), 2);
% 
%                     % determine the target center and range in the regularization windows
%                     for jj = 1:numel(weight_scale)
%                         range_weight(jj,:) = [0, weight_scale(j) - 1] - floor(weight_scale(j) / 2);
%                     end
%                     center_weight = floor((weight_sz + 1)/ 2) + mod(weight_sz + 1,2);
%                     range_h_weight = (center_weight(1)+ range_weight(1,1)) : (center_weight(1) + range_weight(1,2));
%                     range_w_weight = (center_weight(2)+ range_weight(2,1)) : (center_weight(2) + range_weight(2,2));
% 
%                     weight_window{1,1,ii}(range_h_weight,range_w_weight) = params.center;
%                 end
%             end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Model update step
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Extract sample
    % Extract image region for training sample
    sample_pos = round(pos);
    xl = extract_features_new(im, sample_pos, currentScaleFactor_1,currentScaleFactor_2, features, global_fparams, feature_extract_info);
    if seq.frame == 1
        projection_matrix = init_projection_matrix(xl, compressed_dim, params);
    end
    xl_proj = project_sample(xl, projection_matrix);
    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_proj, cos_window, 'uniformoutput', false);
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    
    xlw_r = cellfun(@(feat_map, weight_window) bsxfun(@times, feat_map, weight_window), xlw, weight_window, 'uniformoutput', false);
    xlf_r = cellfun(@fft2, xlw_r, 'uniformoutput', false);
    clear xlw
    % Update appearance model
    if (seq.frame == 1)
        model_xf = xlf;
        model_xf_p = cellfun(@(xlf) zeros(size(xlf), 'single'), xlf, 'UniformOutput', false);
        wf_p = cellfun(@(xlf) zeros(size(xlf), 'single'), xlf, 'UniformOutput', false);
        xlf_p = cellfun(@(xlf) zeros(size(xlf), 'single'), xlf, 'UniformOutput', false);
    else
        model_xf_p = model_xf;
        model_xf = cellfun(@(model_xf, xlf,learning_rate) (1 - learning_rate) * model_xf + learning_rate * xlf, model_xf, xlf,params.learning_rate, 'UniformOutput', false);
        wf_p = wf;
    end
    
    % training the filter
    wf = train_DeepTriCF(params, model_xf, yf, reg_window,...
        model_xf_p, wf_p,xlf,xlf_p,xlf_r);
    
    xlf_p = xlf;
    % Update the scale filter
%     if nScales > 0 && params.use_scale_filter
%         scale_filter = scale_filter_update(im, pos, base_target_sz, currentScaleFactor, scale_filter, params);
%     end
    
    % Update the target size (only used for computing output box)
    target_sz = base_target_sz .* [currentScaleFactor_1,currentScaleFactor_2];
    
    %save position and calculate FPS
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    
    curr_t = toc(frame_tic);
    seq.time = seq.time + curr_t;
    if params.print_screen == 1
        if seq.frame == 1
            fprintf('initialize: %f sec.\n', curr_t);
            fprintf('===================\n');
        else
            fprintf('[%04d/%04d] time: %f\n', seq.frame, seq.num_frames, curr_t);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Visualization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % visualization
    if params.visualization
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if seq.frame == 1  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking');
%             set(fig_handle, 'Position', [100, 100, size(im,2), size(im,1)]);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
            
%             output_name = 'Video_name';
%             opengl software;
%             writer = VideoWriter(output_name, 'MPEG-4');
%             writer.FrameRate = 5;
%             open(writer);
        else
            % Do visualization of the sampled confidence scores overlayed
%             resp_sz = round(img_support_sz*currentScaleFactor*scaleFactors(scale_ind));
%             xs = floor(det_sample_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
%             ys = floor(det_sample_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
            
            % To visualize the continuous scores, sample them 10 times more
            % dense than output_sz.
%             sampled_scores_display = fftshift(sample_fs(scores_fs(:,:,scale_ind), 10*output_sz));
            
            figure(fig_handle);
%             set(fig_handle, 'Position', [100, 100, 100+size(im,2), 100+size(im,1)]);
            imagesc(im_to_show);
            hold on;
%             resp_handle = imagesc(xs, ys, sampled_scores_display); colormap hsv;
%             alpha(resp_handle, 0.5);
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1]);
            hold off;
            
%             axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        end
        
        drawnow
        %         if frame > 1
        %             if frame < inf
        %                 writeVideo(writer, getframe(gcf));
        %             else
        %                 close(writer);
        %             end
        %         end
        %          pause
    end
end
% close(writer);

[seq, results] = get_sequence_results(seq);

if params.disp_fps
    disp(['fps: ' num2str(results.fps)]);
end
