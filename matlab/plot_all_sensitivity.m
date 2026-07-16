function plot_all_sensitivity(in_dir, out_dir)
%PLOT_ALL_SENSITIVITY  Plot every sensitivity JSON export in a directory.
%
%   plot_all_sensitivity() scans ../result/pic (relative to this file) for
%   files matching *sensitivity_*.json — both the centralized exports
%   (sensitivity_<ds>_<param>.json) and the FL exports
%   (fl_sensitivity_<ds>_<reg>_<param>.json) — and renders each via
%   plot_sensitivity, saving the figures next to their JSON.
%
%   plot_all_sensitivity(IN_DIR) scans IN_DIR instead.
%   plot_all_sensitivity(IN_DIR, OUT_DIR) writes all figures into OUT_DIR.

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(in_dir)
        in_dir = fullfile(here, '..', 'result', 'pic');
    end
    if nargin < 2; out_dir = ''; end

    files = dir(fullfile(in_dir, '*sensitivity_*.json'));
    if isempty(files)
        warning('No *sensitivity_*.json files found in %s', in_dir);
        return;
    end

    fprintf('Found %d sensitivity export(s) in %s\n', numel(files), in_dir);
    for k = 1:numel(files)
        jp = fullfile(files(k).folder, files(k).name);
        fprintf('[%d/%d] %s\n', k, numel(files), files(k).name);
        try
            if isempty(out_dir)
                plot_sensitivity(jp);
            else
                plot_sensitivity(jp, out_dir);
            end
        catch ME
            warning('Failed on %s: %s', files(k).name, ME.message);
        end
    end
    fprintf('Done.\n');
end
