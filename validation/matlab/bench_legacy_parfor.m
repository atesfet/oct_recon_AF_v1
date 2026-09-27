function bench_legacy_parfor(yIs, nWorkers)
% Legacy throughput benchmark: runs the unmodified per-plane body of
% yOCTProcessTiledScan (no TIFF writing) in a parfor over yIs.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root, 'Reconstruction_code_legacy', 'myOCT')));
addpath(fullfile(root, 'validation', 'matlab'));
delete(gcp('nocreate')); parpool('local', nWorkers);
tmp = tempname; mkdir(tmp);
t = tic;
parfor k = 1:numel(yIs)
    make_reference_plane(yIs(k), fullfile(tmp, sprintf('p%d.mat', k)));
end
el = toc(t);
fprintf('LEGACY_PARFOR planes=%d workers=%d total=%.1fs per_plane=%.2fs\n', numel(yIs), nWorkers, el, el/numel(yIs));
rmdir(tmp, 's');
end
