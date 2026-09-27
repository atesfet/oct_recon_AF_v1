function make_reference_plane(yI, outMat)
% Reference harness for the Python/GPU port.
% Runs the UNMODIFIED legacy myOCT functions for a single output Y plane
% (a serial copy of the parfor body in yOCTProcessTiledScan.m), times every
% stage and dumps intermediates of one tile to a .mat file.
%
% Usage (from shell):
%   matlab -batch "addpath('validation/matlab'); make_reference_plane(2250,'validation/reference_data/ref_y2250.mat')"
%
% Parameters mirror Demo_ScanAndProcess3D.m as used for 10um_FOV_1.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root, 'Reconstruction_code_legacy', 'myOCT')));

tiledScanInputFolder = [fullfile(root, '10um_FOV_1', 'OCTVolume') '/'];
dispersionQuadraticTerm = 8.962e+07;
focusSigma = 10;
cropZRange_mm = [-0.03 0.04];
outputFilePixelSize_um = 2;
dumpX = 6; dumpZ = 4;       % tile whose intermediates are dumped
cuttoffSigma = 3;

f = load(fullfile(tiledScanInputFolder, 'zChosenFocusPositions.mat'));
focusPositionInImageZpix = f.focusPositionInImageZpix;

json = awsReadJSON([tiledScanInputFolder 'ScanInfo.json']);
octSystem = json.octSystem;
reconstructConfig = {'interpMethod', 'sinc5', ...
    'dispersionQuadraticTerm', dispersionQuadraticTerm, 'n', json.tissueRefractiveIndex};
xCenters = json.xCenters_mm; zDepths = json.zDepths;

%% Geometry (same code path as yOCTProcessTiledScan)
t0 = tic;
[dimOneTile_mm, dimOutput_mm] = yOCTProcessTiledScan_createDimStructure(tiledScanInputFolder, focusPositionInImageZpix);
T.createDimStructure_s = toc(t0);
dimOutput_mm.z.values = (dimOutput_mm.z.values(1)):(outputFilePixelSize_um*1e-3):max(dimOutput_mm.z.values);
zAll = dimOutput_mm.z.values;
zAll(zAll < cropZRange_mm(1) | zAll > cropZRange_mm(2)) = [];
dimOutput_mm.z.values = zAll(:)';
imOutSize = [length(dimOutput_mm.z.values) length(dimOutput_mm.x.values) length(dimOutput_mm.y.values)];

%% Plane loop body
tPlane = tic;
t0 = tic;
[fps, yIInFile] = yOCTProcessTiledScan_getScansFromYFrame(yI, tiledScanInputFolder, focusPositionInImageZpix);
T.getScansFromYFrame_s = toc(t0);

stack = zeros(imOutSize(1:2)); totalWeights = zeros(imOutSize(1:2));
T.load_s = 0; T.fft_s = 0; T.abs_s = 0; T.opc_s = 0; T.interp2_s = 0;
fileI = 1;
for xxI = 1:length(xCenters)
    for zzI = 1:length(zDepths)
        fpTxt = fps{fileI}; fileI = fileI + 1;
        t0 = tic;
        [intFrame, dimFrame] = yOCTLoadInterfFromFile([{fpTxt}, reconstructConfig, ...
            {'dimensions', dimOneTile_mm 'YFramesToProcess', yIInFile, 'octSystem', octSystem}]);
        T.load_s = T.load_s + toc(t0);
        t0 = tic;
        [scanCpx, ~] = yOCTInterfToScanCpx([{intFrame} {dimFrame} reconstructConfig]);
        T.fft_s = T.fft_s + toc(t0);
        t0 = tic;
        scan1 = abs(scanCpx);
        for i = length(size(scan1)):-1:3
            scan1 = squeeze(mean(scan1, i));
        end
        T.abs_s = T.abs_s + toc(t0);
        scanAbsBeforeOPC = scan1;
        t0 = tic;
        [scan1, validMap] = yOCTOpticalPathCorrection(scan1, dimFrame, json);
        T.opc_s = T.opc_s + toc(t0);

        t0 = tic;
        zI = 1:length(dimFrame.z.values); zI = zI(:);
        factorZ = yOCTProcessTiledScan_factorZ(zI, focusPositionInImageZpix(zzI), focusSigma);
        factor = repmat(factorZ, [1 size(scan1, 2)]);
        factor(~validMap) = 0;
        x = dimFrame.x.values + xCenters(xxI);
        z = dimFrame.z.values + zDepths(zzI);
        z = z - dimFrame.z.values(round(focusPositionInImageZpix(zzI)));
        x(1) = x(1) - 1e-10; x(end) = x(end) + 1e-10;
        z(1) = z(1) - 1e-10; z(end) = z(end) + 1e-10;
        [xxAll, zzAll] = meshgrid(dimOutput_mm.x.values, dimOutput_mm.z.values);
        stack = stack + interp2(x, z, scan1.*factor, xxAll, zzAll, 'linear', 0);
        totalWeights = totalWeights + interp2(x, z, factor, xxAll, zzAll, 'linear', 0);
        T.interp2_s = T.interp2_s + toc(t0);

        if xxI == dumpX && zzI == dumpZ
            D.fp = fpTxt;
            D.interf = intFrame;                       % after apodization subtraction
            [D.interfEquispaced, dimEq] = yOCTEquispaceInterf(intFrame, ...
                yOCTChangeDimensionsStructureUnits(dimFrame, 'nm'), 'sinc5');
            D.lambdaEquispaced_nm = dimEq.lambda.values;
            D.scanCpx = scanCpx;
            D.scanAbs = scanAbsBeforeOPC;
            D.scanOPC = scan1;
            D.validMap = validMap;
            D.factor = factor;
            D.x = x; D.z = z;
            D.dimFrame = dimFrame;
        end
    end
end
minFactor1 = exp(-cuttoffSigma^2/2);
totalWeightsRaw = totalWeights;
totalWeights(totalWeights < minFactor1) = NaN;
stackmean = stack ./ totalWeights;
planeDb = mag2db(stackmean);
T.plane_total_s = toc(tPlane);

fprintf('Timing (s) for yI=%d:\n', yI); disp(T);
save(outMat, 'T', 'D', 'stack', 'totalWeightsRaw', 'planeDb', 'yI', 'yIInFile', 'fps', ...
    'dimOneTile_mm', 'dimOutput_mm', 'focusPositionInImageZpix', '-v7.3');
end
