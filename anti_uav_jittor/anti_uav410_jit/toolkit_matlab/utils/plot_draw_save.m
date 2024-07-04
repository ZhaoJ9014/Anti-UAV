function plot_draw_save(num_tracker, plot_style, ave_success_rate_plot, idx_seq_set, rank_num, ...
                        ranking_type, rank_idx, name_tracker_all, threshold_set, title_name, ...
                        x_label_name, y_label_name, fig_name, save_fig_path, save_fig_suf)
% plot and save curves
perf = zeros(1, num_tracker);
for i=1:num_tracker
    %each row is the sr plot of one sequence
    tmp = ave_success_rate_plot(i, idx_seq_set, :);
    aa  = reshape(tmp, [numel(idx_seq_set), size(ave_success_rate_plot, 3)]);
    aa  = aa(sum(aa,2)>eps, :);
    bb  = mean(aa);
    switch ranking_type
        case 'AUC'
            perf(i) = mean(bb);
        case 'threshold'
            perf(i) = bb(rank_idx);
    end
end

[~, index_sort] = sort(perf,'descend');

i = 1;

% plot settings
font_size        = 22; %16
% font_size        = 14;
% font_size_legend = 12;   % for overall plot
font_size_legend = 18;    %8  % for attribute-based plot
axex_font_size   = 14;

tmp_figure = figure('color',[1 1 1]);

set(gcf, 'unit', 'normalized', 'position', [0.2,0.2,0.45,0.6]);      % for overall plot
% set(gcf, 'unit', 'normalized', 'position', [0.01,0.01,0.45,0.88]); % for attribute-based plot

tmp_axes = axes('Parent', tmp_figure, 'FontSize', axex_font_size);
for k = index_sort(1:rank_num)

    tmp = ave_success_rate_plot(k, idx_seq_set, :);
    aa  = reshape(tmp, [numel(idx_seq_set), size(ave_success_rate_plot, 3)]);
    aa  = aa(sum(aa,2)>eps, :);
    bb  = mean(aa);
    
    switch ranking_type
        case 'AUC'
            score = mean(bb);
            tmp   = sprintf('%.3f', score);
        case 'threshold'
            score = bb(rank_idx);
            tmp   = sprintf('%.3f', score);
    end    
    
    tmpName{i} = ['[' tmp '] ' name_tracker_all{k}];
    plot(threshold_set, bb, 'color', plot_style{i}.color, 'lineStyle', plot_style{i}.lineStyle,'lineWidth', 4,'Parent', tmp_axes);
    hold on
    grid on;
    if k == index_sort(1)
%         set(gca,'GridLineStyle', ':', 'GridColor', 'k', 'GridAlpha', 1, 'LineWidth', 1.2);
        set(gca,'GridLineStyle', ':', 'GridColor', 'k', 'GridAlpha', 0.6, 'LineWidth', 1.0);
    end
    i = i + 1;
end


if strcmp(ranking_type, 'threshold')
    legend_position = 'Southeast';  % 'Southeast' or 'Southeastoutside'
else
    legend_position = 'Southwest';  % 'Southwest' or 'Southwestoutside'
end

legend(tmpName, 'Interpreter', 'none', 'fontsize', font_size_legend, 'Location', legend_position);
title(title_name, 'fontsize', font_size);
xlabel(x_label_name, 'fontsize', font_size);
ylabel(y_label_name, 'fontsize', font_size);

hold off

% save result figures
if ~exist(save_fig_path, 'dir')
    mkdir(save_fig_path);
end

% set(gca,'position',[0.1,0.12,0.82,0.8]);
set(gcf,'Units','Inches');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
filename = [save_fig_path fig_name]; % 设定导出文件名
print(gcf,filename,'-dpdf','-r0')
% close(gcf)
% saveas(gcf,[figName,'.pdf']);
 imwrite(frame2im(getframe(gcf)),[save_fig_path fig_name,'.png']);
%saveas(gcf,figName,'png');


% if strcmp(save_fig_suf, 'eps')
%     print('-depsc', [save_fig_path fig_name]);
% else
%     saveas(gcf, [save_fig_path fig_name], 'png');
% end

end