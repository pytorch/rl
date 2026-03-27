@echo off
set TORCHRL_BUILD_VERSION=0.11.0
echo TORCHRL_BUILD_VERSION is set to %TORCHRL_BUILD_VERSION%

@echo on

set VC_VERSION_LOWER=17
set VC_VERSION_UPPER=18
if "%VC_YEAR%" == "2019" (
    set VC_VERSION_LOWER=16
    set VC_VERSION_UPPER=17
)
if "%VC_YEAR%" == "2017" (
    set VC_VERSION_LOWER=15
    set VC_VERSION_UPPER=16
)

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -legacy -products * -version [%VC_VERSION_LOWER%^,%VC_VERSION_UPPER%^) -property installationPath`) do (
    if exist "%%i" if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto vswhere
    )
)

:vswhere
if "%VSDEVCMD_ARGS%" == "" (
    call "%VS15VCVARSALL%" x64 || exit /b 1
) else (
    call "%VS15VCVARSALL%" x64 %VSDEVCMD_ARGS% || exit /b 1
)

@echo on

if "%CU_VERSION%" == "xpu" call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

set DISTUTILS_USE_SDK=1

:: Upgrade setuptools before installing PyTorch
pip install --upgrade setuptools==72.1.0 || exit /b 1

:: Workaround for free-threaded Python on Windows
:: The library is python3XXt.lib but linker expects python3XX.lib
if exist "%CONDA_PREFIX%\libs\python313t.lib" (
    if not exist "%CONDA_PREFIX%\libs\python313.lib" (
        copy "%CONDA_PREFIX%\libs\python313t.lib" "%CONDA_PREFIX%\libs\python313.lib"
    )
)
if exist "%CONDA_PREFIX%\libs\python314t.lib" (
    if not exist "%CONDA_PREFIX%\libs\python314.lib" (
        copy "%CONDA_PREFIX%\libs\python314t.lib" "%CONDA_PREFIX%\libs\python314.lib"
    )
)

set args=%1
shift
:start
if [%1] == [] goto done
set args=%args% %1
shift
goto start

:done
if "%args%" == "" (
    echo Usage: vc_env_helper.bat [command] [args]
    echo e.g. vc_env_helper.bat cl /c test.cpp
)

%args% || exit /b 1
