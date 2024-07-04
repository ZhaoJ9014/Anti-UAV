function [errCoverage, err_center] = calc_seq_err_robust(results, rect_anno, norm_dst)
% calculate center distance error and overlap
% seq_length = results.len;
seq_length = size(rect_anno, 1);

% %%%%%%%%% for monkey-17, we only keep the first 2260 frames (1-2260), and
% we need to cut off for the tracking results
if size(results, 1) ~= size(rect_anno, 1)
    results = results(1:size(rect_anno, 1), :);
end

% handle the invalided tracking results (NAN, negative and even complex ones)
for i = 2:seq_length
    r = results(i,:);
    r_anno = rect_anno(i,:);
     
    if (sum(isnan(r)) | ~isreal(r) | r(3)<=0 | r(4)<=0) & (~isnan(r_anno))
        results(i,:) = results(i-1,:);
    end
end

rect_mat = results;
rect_mat(1,:) = rect_anno(1,:);  % ignore the result in the first frame


% center position
center_GT = [rect_anno(:,1)+(rect_anno(:,3)-1)/2 ...
             rect_anno(:,2)+(rect_anno(:,4)-1)/2];

center = [rect_mat(:,1)+(rect_mat(:,3)-1)/2 ...
          rect_mat(:,2)+(rect_mat(:,4)-1)/2];

% the new seq_length, since we remove the absent frames
new_seq_length = size(rect_anno, 1);
      
% % computer center distance
if norm_dst
    center(:, 1) = center(:, 1)./rect_anno(:, 3);
    center(:, 2) = center(:, 2)./rect_anno(:, 4);
    center_GT(:, 1) = center_GT(:, 1)./rect_anno(:, 3);
    center_GT(:, 2) = center_GT(:, 2)./rect_anno(:, 4);
end
err_center = sqrt(sum(((center(1:new_seq_length,:)-center_GT(1:new_seq_length,:)).^2),2));

index = rect_anno > 0;
idx   = (sum(index, 2)==4);

% calculate overlap
tmp = calc_rect_int(rect_mat(idx,:), rect_anno(idx,:));

errCoverage      = -ones(length(idx),1);
errCoverage(idx) = tmp;
err_center(~idx) = -1;
end
