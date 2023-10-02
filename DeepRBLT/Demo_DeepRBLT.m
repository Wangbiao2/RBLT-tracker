close all
clear
clc

% Add paths
setup_paths();

% Load video information
video_path = './sequences/Animal1';
[seq, ground_truth] = load_video_info(video_path);

% Run DeepRBLT
results = run_DeepRBLT(seq);

close all;