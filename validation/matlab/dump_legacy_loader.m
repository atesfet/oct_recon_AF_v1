function dump_legacy_loader(cases, outMat)
% Dump the output of the UNMODIFIED legacy loaders (yOCTLoadInterfFromFile & friends) for
% single tile folders, for comparison with octrecon.io readers (tests/test_formats.py).
%
% cases: struct array with fields
%   name    - identifier (valid MATLAB field name)
%   folder  - tile folder
%   kind    - 'thorlabs' | 'srr' | 'wasatch' | 'detect'
%   system  - octSystem to pass ('' = let legacy decide)
%   yframe  - YFramesToProcess (1-based)
% For 'srr' / 'wasatch' the header is built with the manufacturer header function directly
% (yOCTLoadInterfFromFile's own header branch is broken for these, see docs/06) and passed
% as 'dimensions'. 'detect' only runs yOCTLoadInterfFromFile_WhatOCTSystemIsIt.
here = fileparts(mfilename('fullpath'));
legacy = fullfile(fileparts(fileparts(fileparts(here))), 'Reconstruction_code_legacy', 'myOCT');
addpath(genpath(legacy));
R = struct();
for i = 1:numel(cases)
    c = cases(i);
    r = struct('ok', true, 'err', '');
    try
        f = [c.folder '/'];
        switch c.kind
            case 'detect'
                [r.octSystem, r.manufacturer] = yOCTLoadInterfFromFile_WhatOCTSystemIsIt(f);
            case 'thorlabs'
                [interf, dim, apod] = yOCTLoadInterfFromFile(f, 'OCTSystem', c.system, 'YFramesToProcess', c.yframe);
                r.interf = interf; r.apod = apod; r.lambda = dim.lambda.values;
            case 'srr'
                % the regular path (header via yOCTLoadInterfFromFile) is tried first to record its outcome
                try
                    yOCTLoadInterfFromFile(f, 'OCTSystem', c.system, 'YFramesToProcess', c.yframe);
                    r.regularPath = 'ok';
                catch ME1
                    r.regularPath = ME1.message;
                end
                dims = yOCTLoadInterfFromFile_ThorlabsSRRHeader(f, c.system, []);
                [interf, dim, apod] = yOCTLoadInterfFromFile(f, 'dimensions', dims, 'YFramesToProcess', c.yframe);
                r.interf = interf; r.apod = apod; r.lambda = dim.lambda.values;
            case 'wasatch'
                try
                    yOCTLoadInterfFromFile(f, 'OCTSystem', 'Wasatch', 'YFramesToProcess', c.yframe);
                    r.regularPath = 'ok';
                catch ME1
                    r.regularPath = ME1.message;
                end
                dims = yOCTLoadInterfFromFile_WasatchHeader(f);
                [interf, dim, apod] = yOCTLoadInterfFromFile(f, 'dimensions', dims, 'OCTSystem', 'Wasatch', ...
                    'YFramesToProcess', c.yframe);
                r.interf = interf; r.apod = apod; r.lambda = dim.lambda.values;
        end
    catch ME
        r.ok = false; r.err = ME.message;
    end
    fprintf('%s: ok=%d %s\n', c.name, r.ok, r.err);
    R.(c.name) = r;
end
save(outMat, '-struct', 'R', '-v7');
end
