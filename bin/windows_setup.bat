@echo off
rem This script will install miniconda locally to the game.exe of the build

set BIN_PATH=%~dp0
set ROOT_PATH=%BIN_PATH%..\
set DQNA_TEXT=DQNAgent
set OKAY_TEXT=Okay!
set DQNA_PATH=%ROOT_PATH%\%DQNA_TEXT%
set MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
set MINICONDA_EXE=miniconda.exe
set MINICONDA_INSTALL_TYPE=JustMe
set MINICONDA_PATH=%ROOT_PATH%\miniconda

echo.

:prepare_directory
echo Preparing Directory:
IF DEFINED MINICONDA_PATH (
  rd /s /q %MINICONDA_PATH% >nul 2>&1
)

IF EXIST %MINICONDA_EXE% (
  DEL /F %MINICONDA_EXE%
)
echo %OKAY_TEXT%
echo.

:download_install_conda
echo Downloading and Installing Miniconda:
curl %MINICONDA_URL% -o %MINICONDA_EXE%
start /wait "" %MINICONDA_EXE% /InstallationType=%MINICONDA_INSTALL_TYPE% /RegisterPython=0 /S /D=%MINICONDA_PATH%
del %MINICONDA_EXE%
echo %OKAY_TEXT%
echo.

:create_conda_env
echo Creating Environment:
call %MINICONDA_PATH%\Scripts\conda env create -f %BIN_PATH%\environment.yml
echo. 
call %MINICONDA_PATH%\Scripts\conda list -n %DQNA_TEXT%
echo.
call %MINICONDA_PATH%\Scripts\conda env list
echo %OKAY_TEXT%
echo.