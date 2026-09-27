function run_legacy_synthetic(volumes, outDir, sevenZipDir)
% Run the UNMODIFIED legacy yOCTProcessTiledScan on synthetic tiled scans written by
% scripts/make_synthetic_volume.py and write one TIFF per volume.
%
%   volumes     cell array of volume folders (each with ScanInfo.json + synthetic_info.json)
%   outDir      output folder; <outDir>/<volumeName>.tiff (+ <volumeName>.err on failure)
%   sevenZipDir optional folder holding a '7z' executable (legacy .oct unzip needs 7-Zip)
%
% Parameters (synthetic_info.json may override pixel_um / dispersion): dispersionQuadraticTerm 8.949e7, focusSigma 10, focusPositionInImageZpix =
% synthetic_info.json focus_pix, outputFilePixelSize_um 2 (= scan pixel size),
% interpMethod sinc5, no crop. A 2-worker pool is used (14 GB RAM machine).
%
% matlab -batch "addpath('validation/matlab'); run_legacy_synthetic({'/tmp/syn/base'}, '/tmp/out')"
here = fileparts(mfilename('fullpath'));
legacy = fullfile(fileparts(fileparts(fileparts(here))), 'Reconstruction_code_legacy', 'myOCT');
addpath(genpath(legacy));
if exist('sevenZipDir', 'var') && ~isempty(sevenZipDir)
    setenv('PATH', [sevenZipDir ':' getenv('PATH')]);
end
if ischar(volumes); volumes = {volumes}; end
if ~exist(outDir, 'dir'); mkdir(outDir); end
delete(gcp('nocreate')); parpool('Processes', 2);
for i = 1:numel(volumes)
    vol = volumes{i};
    [~, name] = fileparts(vol);
    info = jsondecode(fileread(fullfile(vol, 'synthetic_info.json')));
    px = 2; disp2 = 8.949e7;
    if isfield(info, 'pixel_um'); px = info.pixel_um; end
    if isfield(info, 'dispersion'); disp2 = info.dispersion; end
    tif = fullfile(outDir, [name '.tiff']);
    if exist(tif, 'file'); delete(tif); end
    t = tic;
    try
        yOCTProcessTiledScan([vol '/'], {tif}, ...
            'focusPositionInImageZpix', info.focus_pix, 'focusSigma', 10, ...
            'dispersionQuadraticTerm', disp2, 'outputFilePixelSize_um', px, ...
            'interpMethod', 'sinc5', 'v', false);
        fprintf('LEGACY_OK %s %.1fs\n', name, toc(t));
    catch ME
        fprintf('LEGACY_FAIL %s: %s\n', name, ME.message);
        fid = fopen(fullfile(outDir, [name '.err']), 'w');
        fprintf(fid, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
        fclose(fid);
    end
end
delete(gcp('nocreate'));
end
