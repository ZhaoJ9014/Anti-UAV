function eval_tracker(seqs, trackers, eval_type, name_tracker_all, tmp_mat_path, path_anno, rp_all, norm_dst)
% evaluate each tracker
num_tracker = numel(trackers);

threshold_set_overlap = 0:0.05:1;
threshold_set_error   = 0:50;
if norm_dst
    threshold_set_error = threshold_set_error / 100;
end

for i = 1:numel(seqs) % for each sequence
    s    = seqs{i};      % name of sequence
    
    % load GT and the absent flags
    anno        = dlmread([path_anno s '.txt']);
    
    for k = 1:num_tracker  % evaluate each tracker
        t = trackers{k};   % name of tracker
        
        % load tracking result

        try
            res_path=[t.path s '.txt'];
            res = loadjson(res_path).res;
        catch
            res_path=[t.path s '.txt'];
            if strcmp(t.name,'SwinTrack-Tiny')
                res_path=[t.path s '\bounding_box.txt'];
            end
            res = dlmread(res_path);
            
        end
        
        if strcmp(t.datastructure,'2')
            res(:,3:4)=res(:,3:4)-res(:,1:2)+1;
        end

        
%         res = dlmread(res_path);
%         res = load([rp_all t.name '_tracking_result/' s '.txt']);
%         results = load([rp_all t.name '_tracking_result/' s '.txt']);
%         results = load([rp_all s '_' t.name '.mat']);
%         results = results.results;
        fprintf(['evaluating ' t.name ' on ' s ' ...\n']);
        
%         anno     = rect_anno;
        
        success_num_overlap = zeros(1, numel(threshold_set_overlap));
        success_num_err     = zeros(1, numel(threshold_set_error));
        
%         res = results{1};
        
        if isempty(res)
            break;
        end
        
        [err_coverage, err_center] = calc_seq_err_robust(res, anno, norm_dst);
        
        for t_idx = 1:numel(threshold_set_overlap)
            success_num_overlap(1, t_idx) = sum(err_coverage > threshold_set_overlap(t_idx));
        end
        
        for t_idx = 1:length(threshold_set_error)
            success_num_err(1, t_idx) = sum(err_center <= threshold_set_error(t_idx));
        end
        
        len_all = size(anno, 1);  % number of frames in the sequence
        
        ave_success_rate_plot(k, i, :)     = success_num_overlap/(len_all + eps);
        ave_success_rate_plot_err(k, i, :) = success_num_err/(len_all + eps);
    end
end

% save results
if ~exist(tmp_mat_path, 'dir')
    mkdir(tmp_mat_path);
end

dataName1 = [tmp_mat_path 'aveSuccessRatePlot_' num2str(num_tracker) 'alg_overlap_' eval_type '.mat'];
save(dataName1, 'ave_success_rate_plot', 'name_tracker_all');

dataName2 = [tmp_mat_path 'aveSuccessRatePlot_' num2str(num_tracker) 'alg_error_' eval_type '.mat'];
ave_success_rate_plot = ave_success_rate_plot_err;
save(dataName2, 'ave_success_rate_plot', 'name_tracker_all');

end