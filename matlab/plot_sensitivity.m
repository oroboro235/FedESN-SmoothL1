function plot_sensitivity(json_path, out_dir)
%PLOT_SENSITIVITY  Reproduce a FedSL1ESN sensitivity figure from its JSON export.
%
%   plot_sensitivity(JSON_PATH) reads the JSON written by
%   tsc_centralized_sensitivity.py / tsc_fl_sensitivity.py (the save_run()
%   export) and draws a dual-y-axis figure — accuracy (left) and sparsity
%   (right) versus the swept parameter — saving PNG + PDF next to the JSON.
%
%   plot_sensitivity(JSON_PATH, OUT_DIR) writes the figures into OUT_DIR.
%
%   The JSON schema (see the Python save_run()):
%       .dataset .param_name .reg_type .is_integer .is_log
%       .acc_label .sparsity_label .title .baseline_value
%       .settings (all hyperparameters)
%       .results  (array of {param_value, mean_val_acc, std_val_acc,
%                            mean_val_sparsity, std_val_sparsity})

    if nargin < 2 || isempty(out_dir)
        out_dir = fileparts(json_path);
        if isempty(out_dir); out_dir = '.'; end
    end

    data = jsondecode(fileread(json_path));

    % results may decode as a struct array (many values) or a scalar struct
    % (single value); [R.field] handles both.
    R       = data.results;
    xs      = double([R.param_value]).';
    accs    = double([R.mean_val_acc]).';
    acc_std = double([R.std_val_acc]).';
    sps     = double([R.mean_val_sparsity]).';
    sp_std  = double([R.std_val_sparsity]).';

    % Parallel sweeps may emit values out of order — sort by x for clean lines.
    [xs, order] = sort(xs);
    accs = accs(order); acc_std = acc_std(order);
    sps  = sps(order);  sp_std  = sp_std(order);

    colAcc = [0.129 0.400 0.674];   % blue        — accuracy
    colSp  = [0.839 0.376 0.302];   % orange-red  — sparsity

    fig = figure('Color', 'w', 'Position', [100 100 900 500], 'Visible', 'off');
    ax  = gca;

    % ── Left axis: accuracy ──────────────────────────────────────────────────
    yyaxis left
    local_band(xs, accs, acc_std, colAcc);
    hAcc = plot(xs, accs, '-o', 'Color', colAcc, 'LineWidth', 2, ...
                'MarkerFaceColor', colAcc, 'DisplayName', data.acc_label);
    ylim([0 100]);
    ylabel(local_tex(data.acc_label));
    ax.YColor = colAcc;

    % ── Right axis: sparsity ─────────────────────────────────────────────────
    yyaxis right
    local_band(xs, sps, sp_std, colSp);
    hSp = plot(xs, sps, '--s', 'Color', colSp, 'LineWidth', 2, ...
               'MarkerFaceColor', colSp, 'DisplayName', data.sparsity_label);
    ylim([0 100]);
    ylabel(local_tex(data.sparsity_label));
    ax.YColor = colSp;

    % ── X-axis formatting ────────────────────────────────────────────────────
    yyaxis left
    xlabel(local_tex(data.param_name));
    if local_true(data.is_log)
        set(gca, 'XScale', 'log');
    end
    if local_true(data.is_integer)
        % Integer-valued params (n_clients/n_rounds/units/epochs): tick only at
        % the sampled integers so the axis never shows fractional labels.
        xticks(unique(round(xs)));
    end

    % ── Baseline marker (config value) ───────────────────────────────────────
    if isfield(data, 'baseline_value') && ~isempty(data.baseline_value) ...
            && isnumeric(data.baseline_value)
        xline(double(data.baseline_value), ':', ...
              sprintf('config = %g', data.baseline_value), ...
              'Color', [0.4 0.4 0.4], 'LineWidth', 1.2, ...
              'LabelVerticalAlignment', 'bottom');
    end

    % ── Legend + title ───────────────────────────────────────────────────────
    legend([hAcc hSp], 'Location', 'best', 'Interpreter', 'none');
    title({local_tex(data.title); ['Dataset: ' local_tex(data.dataset)]});
    grid on; box on;

    % ── Save PNG + PDF alongside (or into OUT_DIR) ───────────────────────────
    [~, stem] = fileparts(json_path);
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end
    exportgraphics(fig, fullfile(out_dir, [stem '.png']), 'Resolution', 150);
    exportgraphics(fig, fullfile(out_dir, [stem '.pdf']));
    fprintf('Saved -> %s(.png/.pdf)\n', fullfile(out_dir, stem));
    close(fig);
end


function local_band(xs, y, ystd, col)
%LOCAL_BAND  Shaded mean±std ribbon on the current (yy)axis; skipped if std==0.
    if all(ystd == 0); hold on; return; end
    xf = [xs; flipud(xs)];
    yf = [y - ystd; flipud(y + ystd)];
    patch('XData', xf, 'YData', yf, 'FaceColor', col, 'FaceAlpha', 0.15, ...
          'EdgeColor', 'none', 'HandleVisibility', 'off');
    hold on;
end


function s = local_tex(s)
%LOCAL_TEX  Escape underscores so TeX rendering does not subscript labels.
    s = strrep(s, '_', '\_');
end


function tf = local_true(v)
%LOCAL_TRUE  Robust truthiness for JSON booleans (logical or numeric 0/1).
    tf = ~isempty(v) && ((islogical(v) && v) || (isnumeric(v) && v ~= 0));
end
