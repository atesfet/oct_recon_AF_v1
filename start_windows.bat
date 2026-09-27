@echo off
rem OCT reconstruction web app launcher (Windows). Double-click to start.
rem First run: creates the conda env "oct_reconstruction" (or a local .venv), adds GPU
rem support when an NVIDIA driver is present, then starts the app and opens the browser.
setlocal EnableExtensions
cd /d "%~dp0"
if "%OCT_ENV_NAME%"=="" (set "ENV_NAME=oct_reconstruction") else (set "ENV_NAME=%OCT_ENV_NAME%")

set "CONDA="
for %%C in ("%CONDA_EXE%" "%USERPROFILE%\miniforge3\Scripts\conda.exe" "%USERPROFILE%\mambaforge\Scripts\conda.exe" "%USERPROFILE%\miniconda3\Scripts\conda.exe" "%USERPROFILE%\anaconda3\Scripts\conda.exe" "%LOCALAPPDATA%\miniforge3\Scripts\conda.exe" "%ProgramData%\miniforge3\Scripts\conda.exe" "%ProgramData%\Miniconda3\Scripts\conda.exe" "%ProgramData%\Anaconda3\Scripts\conda.exe") do (
  if not defined CONDA if exist "%%~C" set "CONDA=%%~C"
)
if not defined CONDA for /f "delims=" %%C in ('where conda 2^>nul') do if not defined CONDA set "CONDA=%%C"
if not defined CONDA goto :venv

echo [launcher] using conda: %CONDA%
call "%CONDA%" env list | findstr /r /c:"^%ENV_NAME% " >nul
if errorlevel 1 (
  echo [launcher] creating environment %ENV_NAME% ^(first run only, a few minutes^) ...
  call "%CONDA%" env create -n %ENV_NAME% -f environment.yml
  if errorlevel 1 goto :fail
)
call "%CONDA%" run -n %ENV_NAME% python tools\setup_gpu.py --conda "%CONDA%" --env %ENV_NAME%
call "%CONDA%" run -n %ENV_NAME% --no-capture-output python launch.py %*
goto :end

:venv
if "%OCT_VENV%"=="" (set "VENV=%LOCALAPPDATA%\octrecon\venv") else (set "VENV=%OCT_VENV%")
echo [launcher] conda not found - using a Python virtual environment in %VENV%
where py >nul 2>nul
if errorlevel 1 (set "PY=python") else (set "PY=py -3")
if not exist "%VENV%\Scripts\python.exe" %PY% -m venv "%VENV%"
if not exist "%VENV%\Scripts\python.exe" goto :nopython
"%VENV%\Scripts\python.exe" -m pip install -q --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install -q -r requirements.txt
"%VENV%\Scripts\python.exe" tools\setup_gpu.py --pip
"%VENV%\Scripts\python.exe" launch.py %*
goto :end

:nopython
echo Neither conda nor Python 3 was found. Install Miniforge from https://conda-forge.org/download/
:fail
pause
exit /b 1
:end
endlocal
