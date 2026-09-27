function run_legacy_full(outDir)
% Full-volume timing run of the UNMODIFIED legacy reconstruction
% (yOCTProcessTiledScan, parfor over output y planes on the default local pool).
% Parameters = configs/10um_FOV_1.yaml (dispersion 8.949e7 reproduces the legacy TIFF).
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root, 'Reconstruction_code_legacy', 'myOCT')));
volumeOutputFolder = [fullfile(root, '10um_FOV_1', 'OCTVolume') '/'];
if ~exist(outDir, 'dir'); mkdir(outDir); end
outputTiffFile = fullfile(outDir, '10um_FOV_1_recon_matlab.tiff');
f = load(fullfile(volumeOutputFolder, 'zChosenFocusPositions.mat'));
pool = gcp();                     % default local pool (same as legacy usage)
nWorkers = pool.NumWorkers;
t = tic;
yOCTProcessTiledScan(volumeOutputFolder, {outputTiffFile}, ...
    'focusPositionInImageZpix', f.focusPositionInImageZpix, ...
    'focusSigma', 10, 'cropZRange_mm', [-0.03 0.04], ...
    'dispersionQuadraticTerm', 8.949e7, 'outputFilePixelSize_um', 2, ...
    'interpMethod', 'sinc5', 'v', true);
total = toc(t);
fprintf('LEGACY_FULL total=%.1fs (%.2f h) workers=%d\n', total, total/3600, nWorkers);
res.total_seconds = total; res.workers = nWorkers; res.output = outputTiffFile;
res.matlab_version = version; res.finished = datestr(now);
fid = fopen(fullfile(root, 'benchmarks', 'results', 'full_run_matlab.json'), 'w');
fprintf(fid, '%s', jsonencode(res)); fclose(fid);
end
