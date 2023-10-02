% Demo for DeepRBLT++ (ResNet)

function results = tracker_DeepRBLT_Res(params)
[seq, im] = get_sequence_info(params.seq);
params = rmfield(params, 'seq');

if isempty(im)
    seq.rect_position = [];
    [~, results] = get_sequence_results(seq);
    return;
end

seq = get_sequence_vot(seq);

pos = seq.init_pos(:)';
target_sz = seq.init_sz(:)';
params.init_sz = target_sz;
features = params.t_features;
params = init_default_params(params);
init_target_sz = target_sz;
% >

% Set Global parameters and data type
% <
if isfield(params, 't_global')
    global_fparams = params.t_global;
else
    global_fparams = [];
end
global_fparams.use_gpu = params.use_gpu;
global_fparams.gpu_id = params.gpu_id;
global_fparams.augment = 0;

if params.use_gpu
    if isempty(params.gpu_id)
        gD = gpuDevice();
    elseif params.gpu_id > 0
        gD = gpuDevice(params.gpu_id);
    end
    params.data_type = zeros(1, 'single', 'gpuArray');
else
    params.data_type = zeros(1, 'single');
end
params.data_type_complex = complex(params.data_type);

global_fparams.data_type = params.data_type;
% >

% Check if color image
% <
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
% >

if is_color_image
    compressed_dim = [5;20;96];
else
    compressed_dim = [20;96];
end

% Calculate search area and initial scale factor
% <
search_area = prod(params.search_area_scale'*init_target_sz,2);
currentScaleFactor = zeros(numel(search_area),1);
for i = 1 : numel(currentScaleFactor)
    if search_area(i) > params.max_image_sample_size(i)
        currentScaleFactor(i) = sqrt(search_area(i) / params.max_image_sample_size(i));
    elseif search_area(i) < params.min_image_sample_size(i)
        currentScaleFactor(i) = sqrt(search_area(i) / params.min_image_sample_size(i));
    else
        currentScaleFactor(i) = 1.0;
    end
end
% >

% target size at the initial scale
% <
base_target_sz = 1 ./ currentScaleFactor*target_sz;
% >


% search window size
% <
img_sample_sz = repmat(sqrt(prod(base_target_sz,2) .* (params.search_area_scale.^2')),1,2); % square area, ignores the target aspect ratio
% >

% initialise feature settings
% <
[features, global_fparams, feature_info] = init_features(features, global_fparams, params, is_color_image, img_sample_sz, 'odd_cells');
img_support_sz = feature_info.img_support_sz;
feature_sz = feature_info.data_sz;
num_feature_blocks = size(feature_sz, 1);
% >

% Size of the extracted feature maps (filters)
% <
feature_sz_cell = permute(mat2cell(feature_sz, ones(1,num_feature_blocks), 2), [2 3 1]);
filter_sz = feature_sz + mod(feature_sz+1, 2);
filter_sz_cell = permute(mat2cell(filter_sz, ones(1,num_feature_blocks), 2), [2 3 1]);

[output_sz, k1] = max(filter_sz, [], 1);
params.output_sz = output_sz;
k1 = k1(1);
block_inds = 1:num_feature_blocks;
block_inds(k1) = [];
% >

% Pre-computes the grid that is used for socre optimization
% <
ky = circshift(-floor((filter_sz_cell{1}(1) - 1)/2) : ceil((filter_sz_cell{1}(1) - 1)/2), [1, -floor((filter_sz_cell{1}(1) - 1)/2)]);
kx = circshift(-floor((filter_sz_cell{1}(2) - 1)/2) : ceil((filter_sz_cell{1}(2) - 1)/2), [1, -floor((filter_sz_cell{1}(2) - 1)/2)])';
newton_iterations = params.newton_iterations;
% >

% Construct the Gaussian label function, cosine window and initial mask
% <
yf = cell(1,1,num_feature_blocks);
for i = 1:num_feature_blocks
    sz = filter_sz_cell{i};
    output_sigma_factor = params.output_sigma_factor(feature_info.feature_is_deep(i)+1);
    output_sigma  = sqrt(prod(floor(base_target_sz(feature_info.feature_is_deep(i)+1,:))))*feature_sz_cell{i}./img_support_sz{i}* output_sigma_factor;
    rg            = circshift(-floor((sz(1)-1)/2):ceil((sz(1)-1)/2), [0 -floor((sz(1)-1)/2)]);
    cg            = circshift(-floor((sz(2)-1)/2):ceil((sz(2)-1)/2), [0 -floor((sz(2)-1)/2)]);
    [rs, cs]      = ndgrid(rg,cg);
    y             = exp(-0.5 * (((rs.^2 + cs.^2) / mean(output_sigma)^2)));
    yf{i}         = fft2(y);
end
yf = cellfun(@(yf) cast(yf, 'like', params.data_type), yf, 'uniformoutput', false);
clear y
% params.data_type_complex = complex(params.data_type);
global_fparams.data_type = params.data_type;

cos_window = cellfun(@(sz) hann(sz(1)+2)*hann(sz(2)+2)', feature_sz_cell, 'uniformoutput', false);
cos_window = cellfun(@(cos_window) cast(cos_window(2:end-1,2:end-1), 'like', params.data_type), cos_window, 'uniformoutput', false);
params.cos_window = cos_window;

params.feature_downsample_ratio = [4,4,14];
params.weight_downsample_ratio =[4,4,14];

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
    weight_window{1,1,ii} = ones(weight_sz) * params.background;
    range_weight = zeros(numel(weight_scale), 2);
    
    % determine the target center and range in the regularization windows
    for jj = 1:numel(weight_scale)
        range_weight(jj,:) = [0, weight_scale(j) - 1] - floor(weight_scale(j) / 2);
    end
    center_weight = floor((weight_sz + 1)/ 2) + mod(weight_sz + 1,2);
    range_h_weight = (center_weight(1)+ range_weight(1,1)) : (center_weight(1) + range_weight(1,2));
    range_w_weight = (center_weight(2)+ range_weight(2,1)) : (center_weight(2) + range_weight(2,2));
    
    weight_window{1,1,ii}(range_h_weight,range_w_weight) = params.target;
end

reg_window = cellfun(@(reg_window) cast(reg_window, 'like', params.data_type), reg_window, 'uniformoutput', false);
weight_window = cellfun(@(weight_window) cast(weight_window, 'like', params.data_type), weight_window, 'uniformoutput', false);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use the pyramid filters to estimate the scale
% <
nScales = params.number_of_scales;
scale_step = params.scale_step;
scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
scaleFactors = scale_step .^ scale_exp;
% >

seq.time = 0;
scores_fs_feat = cell(1,1,num_feature_blocks);
response_feat = cell(1,1,num_feature_blocks);

while true
    % Read image and timing
    % <
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
    
    tic();
    % >
    
    %% Target localization step
    if seq.frame > 1
        global_fparams.augment = 0;
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            % Extract features at multiple resolutions
            % <
            sample_pos = round(pos);
            sample_scale = currentScaleFactor*scaleFactors;
            [xt, ~] = extract_features(im, sample_pos, sample_scale, features, global_fparams, feature_info);
            xt_proj = project_sample(xt, projection_matrix);
            xtw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xt_proj, cos_window, 'uniformoutput', false);
            xtf = cellfun(@fft2, xtw, 'uniformoutput', false);
            
            % Calculate and fuse responses
            response_handcrafted = 0;
            response_deep = 0;
            for k = [k1 block_inds]
                if feature_info.feature_is_deep(k) == 0
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(wf{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                    response_feat{k} = ifft2(scores_fs_feat{k}, 'symmetric');
                    response_handcrafted = response_handcrafted + response_feat{k};
                else
                    scores_fs_feat{k} = gather(sum(bsxfun(@times, conj(wf{k}), xtf{k}), 3));
                    scores_fs_feat{k} = resizeDFT2(scores_fs_feat{k}, output_sz);
                    response_feat{k} = ifft2(scores_fs_feat{k}, 'symmetric');
                    response_feat{k}(ceil(output_sz(1)/2)+1:output_sz(1)-floor(output_sz(1)/2),:,:,:)=[];
                    response_feat{k}(:,ceil(output_sz(2)/2)+1:output_sz(2)-floor(output_sz(2)/2),:,:)=[];
                    response_deep = response_deep + response_feat{k};
                end
            end
            
            [disp_row, disp_col, sind, ~, response, ~] = resp_newton(squeeze(response_handcrafted)/feature_info.feature_hc_num, squeeze(response_deep)/feature_info.feature_deep_num,...
                newton_iterations, ky, kx, output_sz);
            
            % >
            
            % Compute the translation vector
            % <
            translation_vec = [disp_row, disp_col] .* (img_support_sz{k1}./output_sz) * currentScaleFactor(1) * scaleFactors(sind);
            if seq.frame < 3
                scale_change_factor = scaleFactors(ceil(params.number_of_scales/2));
            else
                scale_change_factor = scaleFactors(sind);
            end
            % >
            
            % update position
            % <
            old_pos = pos;
            if sum(isnan(translation_vec))
                pos = sample_pos;
            else
                pos = sample_pos + translation_vec;
            end
            
            if params.clamp_position
                pos = max([1 1], min([size(im,1) size(im,2)], pos));
            end
            % >
            
            % Update the scale
            % <
            currentScaleFactor = currentScaleFactor * scale_change_factor;
            % >
            iter = iter + 1;
        end
    end
    
    %% Model update step
    % Extract features and learn filters
    % <
    global_fparams.augment = 1;
    sample_pos = round(pos);
    [xl, ~] = extract_features(im, sample_pos, currentScaleFactor, features, global_fparams, feature_info);
    if seq.frame == 1
        projection_matrix = init_projection_matrix(xl, compressed_dim, params);
    end
    xl_proj = project_sample(xl, projection_matrix);

    xlw = cellfun(@(feat_map, cos_window) bsxfun(@times, feat_map, cos_window), xl_proj, cos_window, 'uniformoutput', false);
    xlf = cellfun(@fft2, xlw, 'uniformoutput', false);
    
    xlw_r = cellfun(@(feat_map, weight_window) bsxfun(@times, feat_map, weight_window), xlw, weight_window, 'uniformoutput', false);
    xlf_r = cellfun(@fft2, xlw_r, 'uniformoutput', false);
    clear xlw xlw_r
%     [filter_model_f,spatial_units ] = train_filter(xlf, feature_info, yf, seq, params, filter_model_f);
    
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


    wf = train_DeepRBLT(params, model_xf, yf, reg_window,...
                                        model_xf_p, wf_p,xlf,xlf_p,xlf_r);
    xlf_p = xlf;
    %>
    
    % Update the target size
    % <
    target_sz = base_target_sz(1,:) * currentScaleFactor(1);
    % >
    
    %save position and time
    % <
    tracking_result.center_pos = double(pos);
    tracking_result.target_size = double(target_sz);
    seq = report_tracking_result(seq, tracking_result);
    seq.time = seq.time + toc();
    % >
    
    % visualisation
    % <
    if params.vis_res
        rect_position_vis = [pos([2,1]) - (target_sz([2,1]) - 1)/2, target_sz([2,1])];
        im_to_show = double(im)/255;
        if size(im_to_show,3) == 1
            im_to_show = repmat(im_to_show, [1 1 3]);
        end
        if seq.frame == 1  %first frame, create GUI
            fig_handle = figure('Name', 'Tracking','Position',[100, 300, 600, 480]);
            set(gca, 'position', [0 0 1 1 ]);
            axis off;axis image;
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1], 'FontSize',20);
            hold off;
            axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
        else

            figure(fig_handle);
            imagesc(im_to_show);
            hold on;
            rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
            text(10, 10, int2str(seq.frame), 'color', [0 1 1], 'FontSize',20);
            hold off;
            
        end
        drawnow
    end
    % >
    
end
% get tracking results
% <
[~, results] = get_sequence_results(seq); 
if params.vis_res&& params.vis_details
    close(fig_handle_detail);close(fig_handle);
elseif params.vis_res
    close(fig_handle);
end
% >
