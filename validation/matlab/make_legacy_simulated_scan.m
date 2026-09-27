function make_legacy_simulated_scan(outFolder, nx, ny)
% Write a 'Simulated Ganymede' tiled scan with the UNMODIFIED legacy simulator
% (Simulation/yOCTSimulateTileScan.m, as used by Processing/test_yOCTProcessTiledScan.m)
% plus a synthetic_info.json so run_legacy_synthetic.m / dump_synthetic_plane.m and the
% Python comparison use the same parameters (focus 256, 1 um pixels, dispersion 0).
here = fileparts(mfilename('fullpath'));
legacy = fullfile(fileparts(fileparts(fileparts(here))), 'Reconstruction_code_legacy', 'myOCT');
addpath(genpath(legacy));
yOCTHardware('init', 'OCTSystem', 'Ganymede', 'skipHardware', true);
if ~exist('nx', 'var'); nx = 40; end
if ~exist('ny', 'var'); ny = 4; end
data = zeros(512, nx, ny) + 1;
data([60 150 300], :, :) = 100;
data(200, 1:round(nx/2), :) = 300;
data(230, :, :) = repmat(reshape(50 * (1:ny) / ny, 1, 1, ny), 1, nx, 1);
focus = 256;
yOCTSimulateTileScan(data, [outFolder '/'], 'pixelSize_um', 1, 'zDepths', [0 0.02], ...
    'focusPositionInImageZpix', focus, 'focusSigma', 40, ...
    'octProbePath', yOCTGetProbeIniPath('40x', 'OCTP900', 'SUMMER'));
info = struct('format', 'simulated', 'focus_pix', focus, 'pixel_um', 1, 'dispersion', 0);
fid = fopen(fullfile(outFolder, 'synthetic_info.json'), 'w'); fprintf(fid, '%s', jsonencode(info)); fclose(fid);
end
