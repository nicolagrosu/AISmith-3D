@echo off&&cd /D %~dp0&&chcp 65001>nul
setlocal enabledelayedexpansion
set "node_name=SkinTokens"
Title '%node_name%' for 'ComfyUI Easy Install' by PixelArtistry
:: PixelArtistry Community Edition ::

set "DIR_LVL=..\"
call :SET_COLORS
call :CHECK_FOLDER "ComfyUI-Easy-Install\Add-ons"
call :CHECK_INUSE "Start ComfyUI.bat"

:: Resolve absolute paths to avoid pushd/popd issues ::
pushd "%DIR_LVL%." >nul 2>&1
set "ROOT_DIR=!CD!"
popd >nul 2>&1
set "SKINTOKENS_DIR=!ROOT_DIR!\ComfyUI\custom_nodes\ComfyUI-SkinTokens"
set "PYTHON_EXE=!ROOT_DIR!\python_embeded\python.exe"

call :GET_VERSIONS "3.12" "2.8" "12.8"

set "PIPargs=--no-cache-dir --no-warn-script-location --timeout=1000 --retries 20 --use-pep517"

:: Erasing ~* folders ::
if exist "!ROOT_DIR!\python_embeded\Lib\site-packages\~*" (powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem '!ROOT_DIR!\python_embeded\Lib\site-packages\' -Directory | Where-Object {$_.Name -like '~*'} | Remove-Item -Recurse -Force")

:: Skip downloading LFS (Large File Storage) files ::
set GIT_LFS_SKIP_SMUDGE=1

:: ======================================================================== ::
::                         BLENDER PATH SETUP                               ::
:: ======================================================================== ::
echo.
echo %clGreen%::::::::::::::: Checking%clYellow% Blender%clReset%
echo.

:: Check if Blender is already on PATH ::
where blender >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    for /f "tokens=*" %%B in ('where blender') do set "BLENDER_FOUND=%%B"
    echo %clGreen%  Found on PATH:%clReset% !BLENDER_FOUND!
    goto :BLENDER_VERSION_CHECK
)

:: Search common install locations (picks latest version due to loop overwrite) ::
set "BLENDER_EXE="

:: Program Files (standard installer) ::
for /d %%D in ("C:\Program Files\Blender Foundation\Blender*") do (
    if exist "%%D\blender.exe" set "BLENDER_EXE=%%D"
)

:: Program Files x86 (rare but possible) ::
if not defined BLENDER_EXE (
    for /d %%D in ("C:\Program Files (x86)\Blender Foundation\Blender*") do (
        if exist "%%D\blender.exe" set "BLENDER_EXE=%%D"
    )
)

:: Local AppData (Microsoft Store / per-user install) ::
if not defined BLENDER_EXE (
    for /d %%D in ("%LOCALAPPDATA%\Blender Foundation\Blender*") do (
        if exist "%%D\blender.exe" set "BLENDER_EXE=%%D"
    )
)

:: Steam (common library locations) ::
if not defined BLENDER_EXE (
    for %%S in (
        "C:\Program Files (x86)\Steam\steamapps\common\Blender"
        "D:\SteamLibrary\steamapps\common\Blender"
        "E:\SteamLibrary\steamapps\common\Blender"
    ) do (
        if exist "%%~S\blender.exe" set "BLENDER_EXE=%%~S"
    )
)

:: Other common drive letters (custom installs) ::
if not defined BLENDER_EXE (
    for %%L in (D E F G) do (
        for /d %%D in ("%%L:\Blender*") do (
            if exist "%%D\blender.exe" set "BLENDER_EXE=%%D"
        )
        for /d %%D in ("%%L:\Program Files\Blender Foundation\Blender*") do (
            if exist "%%D\blender.exe" set "BLENDER_EXE=%%D"
        )
    )
)

if not defined BLENDER_EXE (
    echo %clRed%  Blender not found in common locations.%clReset%
    echo %clYellow%  Please install Blender 4.2+ from https://www.blender.org/download/%clReset%
    echo %clYellow%  Then re-run this installer.%clReset%
    echo.
    echo %clYellow%  Press any key to exit...%clReset%&&pause>nul&&exit
)

echo %clGreen%  Found:%clReset% !BLENDER_EXE!\blender.exe
set "BLENDER_FOUND=!BLENDER_EXE!\blender.exe"

:: Add to user PATH permanently (does not require admin) ::
echo %clYellow%  Adding Blender to user PATH...%clReset%
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=[Environment]::GetEnvironmentVariable('Path','User'); if($p -and $p -like '*Blender*'){exit 0}else{exit 1}" >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo %clGreen%  Blender directory already in user PATH.%clReset%
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=[Environment]::GetEnvironmentVariable('Path','User'); if($p){$p=$p+';!BLENDER_EXE!'}else{$p='!BLENDER_EXE!'}; [Environment]::SetEnvironmentVariable('Path',$p,'User')" >nul 2>&1
    set "PATH=!PATH!;!BLENDER_EXE!"
    echo %clGreen%  Added to user PATH.%clReset%
)

:BLENDER_VERSION_CHECK
:: Check Blender version (minimum 4.2 required) ::
set "BLENDER_VER="
"!BLENDER_FOUND!" --version > "%TEMP%\_stblender.tmp" 2>nul
if exist "%TEMP%\_stblender.tmp" (
    for /f "tokens=2 delims= " %%v in ('findstr /R "^Blender" "%TEMP%\_stblender.tmp"') do set "BLENDER_VER=%%v"
    del "%TEMP%\_stblender.tmp" >nul 2>&1
)
if defined BLENDER_VER (
    for /f "tokens=1,2 delims=." %%a in ("!BLENDER_VER!") do (
        set "BL_MAJOR=%%a"
        set "BL_MINOR=%%b"
    )
    if !BL_MAJOR! LSS 4 (
        echo %clRed%  Blender !BLENDER_VER! is too old. SkinTokens requires 4.2 or newer.%clReset%
        echo %clYellow%  Please update from https://www.blender.org/download/%clReset%
        echo.
        echo %clYellow%  Press any key to exit...%clReset%&&pause>nul&&exit
    )
    if !BL_MAJOR!==4 if !BL_MINOR! LSS 2 (
        echo %clRed%  Blender !BLENDER_VER! is too old. SkinTokens requires 4.2 or newer.%clReset%
        echo %clYellow%  Please update from https://www.blender.org/download/%clReset%
        echo.
        echo %clYellow%  Press any key to exit...%clReset%&&pause>nul&&exit
    )
    echo %clGreen%  Blender !BLENDER_VER! OK%clReset%
) else (
    echo %clYellow%  Could not verify Blender version. Proceeding anyway.%clReset%
)

:BLENDER_DONE
echo.

:: ======================================================================== ::
::                       INSTALL COMFYUI-SKINTOKENS                         ::
:: ======================================================================== ::
echo %clGreen%::::::::::::::: Installing%clYellow% ComfyUI-%node_name%%clReset%
echo.
if exist "!SKINTOKENS_DIR!" (
    echo %clYellow%  Existing installation found. Updating...%clReset%
    pushd "!SKINTOKENS_DIR!"
    git.exe pull
    popd
) else (
    git.exe clone https://github.com/Aero-Ex/ComfyUI-SkinTokens.git "!SKINTOKENS_DIR!"
)
echo.

:: ======================================================================== ::
::                         RUN INSTALL.PY                                    ::
:: ======================================================================== ::
echo %clGreen%::::::::::::::: Running%clYellow% install.py%clReset%
echo.
if not exist "!SKINTOKENS_DIR!" (
    echo %clRed%  ERROR: SkinTokens directory not found at: !SKINTOKENS_DIR!%clReset%
    goto :SKIP_INSTALL
)
pushd "!SKINTOKENS_DIR!"
"!PYTHON_EXE!" install.py
popd
:SKIP_INSTALL
echo.

:: ======================================================================== ::
::                      FLASH ATTENTION (FALLBACK)                          ::
:: ======================================================================== ::
echo %clGreen%::::::::::::::: Checking%clYellow% Flash Attention%clReset%
echo.

:: Check if flash_attn is already installed ::
"!PYTHON_EXE!" -c "import flash_attn; print(f'  flash_attn {flash_attn.__version__} already installed')" 2>nul
if !ERRORLEVEL! EQU 0 (
    echo %clGreen%  Flash Attention OK.%clReset%
    goto :FA_DONE
)

:: Check GPU compute capability (Flash Attention 2 requires SM 80+ / Ampere+) ::
set "GPU_SUPPORTED=0"
set "GPU_NAME=Unknown"
set "GPU_CC=unknown"
"!PYTHON_EXE!" -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}'); print(torch.cuda.get_device_name())" > "%TEMP%\_stgpu.tmp" 2>nul
if exist "%TEMP%\_stgpu.tmp" (
    set "_gpuline=0"
    for /f "usebackq delims=" %%a in ("%TEMP%\_stgpu.tmp") do (
        if !_gpuline!==0 (
            set "GPU_CC=%%a"
            for /f "tokens=1 delims=." %%m in ("%%a") do (
                if %%m GEQ 8 set "GPU_SUPPORTED=1"
            )
        )
        if !_gpuline!==1 set "GPU_NAME=%%a"
        set /a "_gpuline+=1"
    )
    del "%TEMP%\_stgpu.tmp" >nul 2>&1
)

if "!GPU_SUPPORTED!"=="0" (
    echo %clYellow%  GPU: !GPU_NAME! ^(SM !GPU_CC!^)%clReset%
    echo %clYellow%  Flash Attention 2 requires Ampere ^(RTX 30-series^) or newer.%clReset%
    echo %clYellow%  Skipping — SkinTokens will use standard PyTorch attention instead.%clReset%
    echo %clYellow%  This works fine, just slightly slower inference.%clReset%
    goto :FA_DONE
)

echo %clGray%  GPU: !GPU_NAME! ^(SM !GPU_CC!^) — Flash Attention supported%clReset%

:: Install from kingbri1 fork (cu128 + torch2.8 + cp312 Windows) ::
echo %clYellow%  Installing Flash Attention 2.8.3...%clReset%
"!PYTHON_EXE!" -m pip install https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp312-cp312-win_amd64.whl %PIPargs%
echo.

:FA_DONE
echo.

:: ======================================================================== ::
::                    DOWNLOAD MODEL CHECKPOINTS                            ::
:: ======================================================================== ::
echo %clGreen%::::::::::::::: Downloading%clYellow% SkinTokens Model%clReset%
echo.
if not exist "!SKINTOKENS_DIR!" (
    echo %clRed%  ERROR: SkinTokens directory not found at: !SKINTOKENS_DIR!%clReset%
    goto :SKIP_DOWNLOAD
)

:: Check if models already exist ::
set "MODELS_EXIST=1"
if not exist "!SKINTOKENS_DIR!\experiments\skin_vae_2_10_32768\last.ckpt" (
    echo %clGray%  Missing: skin_vae checkpoint%clReset%
    set "MODELS_EXIST=0"
)
if not exist "!SKINTOKENS_DIR!\experiments\articulation_xl_quantization_256_token_4\grpo_1400.ckpt" (
    echo %clGray%  Missing: articulation checkpoint%clReset%
    set "MODELS_EXIST=0"
)
if not exist "!SKINTOKENS_DIR!\models\Qwen3-0.6B\config.json" (
    echo %clGray%  Missing: Qwen3-0.6B LLM%clReset%
    set "MODELS_EXIST=0"
)

if "!MODELS_EXIST!"=="1" (
    echo %clGreen%  All model checkpoints already downloaded. Skipping.%clReset%
    goto :SKIP_DOWNLOAD
)

pushd "!SKINTOKENS_DIR!"
"!PYTHON_EXE!" download.py --model
popd
:SKIP_DOWNLOAD
echo.

:: ======================================================================== ::
::                         RESTORE NUMPY                                    ::
:: ======================================================================== ::
:: Ensure numpy stays at 1.26.4 (some deps try to upgrade it) ::
"!PYTHON_EXE!" -c "import numpy, sys; sys.exit(0 if numpy.__version__ == '1.26.4' else 1)" 2>nul || (
    echo %clYellow%  Restoring numpy 1.26.4...%clReset%
    "!PYTHON_EXE!" -m pip install --force-reinstall numpy==1.26.4 --no-deps --no-warn-script-location
)

:: ======================================================================== ::
::                           FINAL MESSAGES                                 ::
:: ======================================================================== ::
echo.
echo %clGreen%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::%clReset%
echo %clGreen%:::::::::::::::%clYellow% %node_name% %clGreen%Installation Complete              ::::::%clReset%
echo %clGreen%::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::%clReset%
echo.
echo %clWhite%  Notes:%clReset%
echo %clGray%  - Blender 4.2+ must be on PATH (this script handles it)%clReset%
echo %clGray%  - Set bpy_server_mode to "Headless (Blender)" in the node%clReset%
echo %clGray%  - Model checkpoint: skintoken/experiments/...grpo_1400.ckpt%clReset%
echo.
if "%WARNINGS%"=="1" (
    echo %clYellow%  Some warnings occurred. Check the output above.%clReset%
    echo.
)

if "%~1"=="" (
    echo %clGreen%::::::::::::::: %clYellow%Press any key to exit%clReset%&Pause>nul
    exit
)
exit /b

:: ======================================================================== ::
::                          HELPER FUNCTIONS                                 ::
:: ======================================================================== ::

:SET_COLORS
for /f %%a in ('"prompt $E & for %%b in (1) do rem"') do set "ESC=%%a"
set "clWarn=%ESC%[33m"
set "clGray=%ESC%[90m"
set "clRed=%ESC%[91m"
set "clGreen=%ESC%[92m"
set "clYellow=%ESC%[93m"
set "clBlue=%ESC%[94m"
set "clMagenta=%ESC%[95m"
set "clCyan=%ESC%[96m"
set "clWhite=%ESC%[97m"
set "clReset=%ESC%[0m"
GOTO :EOF

:CHECK_INUSE
set "StartComfyUI=%DIR_LVL%%~1"
set "path=%windir%\System32;%windir%\System32\WindowsPowerShell\v1.0;%localappdata%\Microsoft\WindowsApps;%PATH%"
if exist "%StartComfyUI%" (
    set PORT=8188
    for /f %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "([regex]::Match((Get-Content '%StartComfyUI%' -Raw), '--port\s+(\d+)')).Groups[1].Value"') do set PORT=%%A
    for /f %%A in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Get-NetTCPConnection -LocalPort !PORT! -State Listen -ErrorAction SilentlyContinue) { 1 } else { 0 }"') do set INUSE=%%A
    if "!INUSE!"=="1" (
        echo.
        echo    %clWhite%ComfyUI%clReset% is already running on port %clGreen%!PORT!%clReset%. %clWhite%Please close it first.%clReset%
        echo.
        echo    %clGray%Press any key to exit...%clReset%&&pause>nul&&exit
    )
)
GOTO :EOF

:CHECK_FOLDER
set "PYTHON_EXE="
set "PREF_FOLDER=%~1"

if exist "%DIR_LVL%python_embeded\python.exe" (set "PYTHON_EXE=%DIR_LVL%python_embeded\python.exe")

if "!PYTHON_EXE!"=="" (
    echo.
    echo    %clGreen%Please run this file from the %clYellow%%~1%clGreen% folder.%clReset%
    echo.
    echo    %clGray%Press any key to exit...%clReset%&&pause>nul&&exit
)
GOTO :EOF

:GET_VERSIONS
set "WARNINGS=0"
set "req_python=%~1"
set "req_torch=%~2"
set "req_cuda=%~3"

:: Python version ::
set "PYVER=Not found"
for /f "tokens=2 delims= " %%a in ('"%PYTHON_EXE%" --version 2^>nul') do set "PYVER=%%a"
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do set "PYVER_SHORT=%%a.%%b"
call :CHECK_VERSION "Python" "%PYVER_SHORT%" "%req_python%"

:: Torch and CUDA versions (temp file avoids for /f quoting issues) ::
set "TORCHVER=Not found"
set "CUDAVER=Not available"
"%PYTHON_EXE%" -c "import torch; print(torch.__version__.split('+')[0]); print(torch.version.cuda)" > "%TEMP%\_stver.tmp" 2>nul
if exist "%TEMP%\_stver.tmp" (
    set "_linenum=0"
    for /f "usebackq delims=" %%a in ("%TEMP%\_stver.tmp") do (
        if !_linenum!==0 set "TORCHVER=%%a"
        if !_linenum!==1 set "CUDAVER=%%a"
        set /a "_linenum+=1"
    )
    del "%TEMP%\_stver.tmp" >nul 2>&1
)
for /f "tokens=1,2 delims=." %%a in ("%TORCHVER%") do set "TORCHVER_SHORT=%%a.%%b"
call :CHECK_VERSION "Torch" "%TORCHVER_SHORT%" "%req_torch%"
for /f "tokens=1,2 delims=." %%a in ("%CUDAVER%") do set "CUDAVER_SHORT=%%a.%%b"
call :CHECK_VERSION "CUDA" "%CUDAVER_SHORT%" "%req_cuda%"

echo.
echo    %clGray%Python %PYVER% ^| Torch %TORCHVER% ^| CUDA %CUDAVER%%clReset%
echo.
GOTO :EOF

:CHECK_VERSION
set "DISPLAY=%~1"
set "CURRENT=%~2"
set "ALLOWED=%~3"
set "FOUND=0"

if "%CURRENT%"=="" (
    echo %clWarn%WARNING: %clRed%%DISPLAY% version not detected.%clReset%
    set "WARNINGS=1"
    GOTO :EOF
)
if "%CURRENT%"=="Not available" (
    echo %clWarn%WARNING: %clRed%%DISPLAY% is not available.%clReset%
    set "WARNINGS=1"
    GOTO :EOF
)
if "%CURRENT%"=="Not found" (
    echo %clWarn%WARNING: %clRed%%DISPLAY% is not found.%clReset%
    set "WARNINGS=1"
    GOTO :EOF
)

for %%v in (%ALLOWED%) do (
    if "%CURRENT%"=="%%v" set "FOUND=1"
)

if "%FOUND%"=="0" (
    echo %clWarn%WARNING: %clRed%%DISPLAY% %CURRENT% is not supported. %clGreen%Supported: %ALLOWED%%clReset%
    set "WARNINGS=1"
)
GOTO :EOF
