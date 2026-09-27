function dump_synthetic_plane(vol, yI, outMat)
% Serial copy of the yOCTProcessTiledScan.m parfor body (l.262-375) for one output y plane of
% a synthetic volume (same parameters as run_legacy_synthetic.m; synthetic_info.json fields
% focus_pix, optional pixel_um / dispersion), calling only UNMODIFIED
% legacy functions. Saves stack, totalWeights (before NaN) and planeDb (float, unquantised).
here = fileparts(mfilename('fullpath'));
legacy = fullfile(fileparts(fileparts(fileparts(here))), 'Reconstruction_code_legacy', 'myOCT');
addpath(genpath(legacy));
vol = [vol '/'];
info = jsondecode(fileread([vol 'synthetic_info.json']));
json = awsReadJSON([vol 'ScanInfo.json']);
focusSigma = 10; cuttoffSigma = 3; outputFilePixelSize_um = 2; disp2 = 8.949e7;
if isfield(info, 'pixel_um'); outputFilePixelSize_um = info.pixel_um; end
if isfield(info, 'dispersion'); disp2 = info.dispersion; end
reconstructConfig = {'interpMethod', 'sinc5', 'dispersionQuadraticTerm', disp2, 'n', json.tissueRefractiveIndex};
focusPositionInImageZpix = info.focus_pix * ones(1, length(json.zDepths));
xCenters = json.xCenters_mm; zDepths = json.zDepths; octSystem = json.octSystem;
[dimOneTile_mm, dimOutput_mm] = yOCTProcessTiledScan_createDimStructure(vol, focusPositionInImageZpix);
dimOutput_mm.z.values = (dimOutput_mm.z.values(1)):(outputFilePixelSize_um*1e-3):max(dimOutput_mm.z.values);
imOutSize = [length(dimOutput_mm.z.values) length(dimOutput_mm.x.values)];
[fps, yIInFile] = yOCTProcessTiledScan_getScansFromYFrame(yI, vol, focusPositionInImageZpix);
stack = zeros(imOutSize); totalWeights = zeros(imOutSize);
fileI = 1;
for xxI = 1:length(xCenters)
    for zzI = 1:length(zDepths)
        fpTxt = fps{fileI}; fileI = fileI + 1;
        [intFrame, dimFrame] = yOCTLoadInterfFromFile([{fpTxt}, reconstructConfig, ...
            {'dimensions', dimOneTile_mm 'YFramesToProcess', yIInFile, 'octSystem', octSystem}]);
        [scan1, ~] = yOCTInterfToScanCpx([{intFrame} {dimFrame} reconstructConfig]);
        scan1 = abs(scan1);
        for i = length(size(scan1)):-1:3
            scan1 = squeeze(mean(scan1, i));
        end
        tiles.(sprintf('t%d_%d', xxI, zzI)) = scan1;
        [scan1, validMap] = yOCTOpticalPathCorrection(scan1, dimFrame, json);
        zI = 1:length(dimFrame.z.values); zI = zI(:);
        factor = repmat(yOCTProcessTiledScan_factorZ(zI, focusPositionInImageZpix(zzI), focusSigma), [1 size(scan1, 2)]);
        factor(~validMap) = 0;
        x = dimFrame.x.values + xCenters(xxI);
        z = dimFrame.z.values + zDepths(zzI);
        z = z - dimFrame.z.values(round(focusPositionInImageZpix(zzI)));
        x(1) = x(1) - 1e-10; x(end) = x(end) + 1e-10;
        z(1) = z(1) - 1e-10; z(end) = z(end) + 1e-10;
        [xxAll, zzAll] = meshgrid(dimOutput_mm.x.values, dimOutput_mm.z.values);
        stack = stack + interp2(x, z, scan1.*factor, xxAll, zzAll, 'linear', 0);
        totalWeights = totalWeights + interp2(x, z, factor, xxAll, zzAll, 'linear', 0);
        valid.(sprintf('t%d_%d', xxI, zzI)) = validMap;
    end
end
totalWeightsRaw = totalWeights;
totalWeights(totalWeights < exp(-cuttoffSigma^2/2)) = NaN;
planeDb = mag2db(stack ./ totalWeights);
zOut = dimOutput_mm.z.values; xOut = dimOutput_mm.x.values;
save(outMat, 'stack', 'totalWeightsRaw', 'planeDb', 'tiles', 'valid', 'zOut', 'xOut', 'yIInFile', '-v7');
end
