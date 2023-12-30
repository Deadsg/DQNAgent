@echo off
rem This script will install miniconda locally to the game.exe of the build
rem a: K-Rawson y:2024

set BIN_PATH=%~dp0
set ROOT_PATH=%BIN_PATH%..\
set DQNA_PATH=%ROOT_PATH%\DQNAgent
set MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
set MINICONDA_EXE=miniconda.exe
set MINICONDA_PATH=%ROOT_PATH%\miniconda

echo.
echo Preparing Directory:
IF DEFINED MINICONDA_PATH (
  rd /s /q %MINICONDA_PATH% >nul 2>&1
)

IF EXIST %MINICONDA_EXE% (
  DEL /F %MINICONDA_EXE%
)
echo Okay!
echo.

echo Downloading and Installing Miniconda:
curl %MINICONDA_URL% -o %MINICONDA_EXE%
start /wait "" %MINICONDA_EXE% /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_PATH%
del %MINICONDA_EXE%
echo Okay!
echo.

echo Creating Environment:
%MINICONDA_PATH%\Scripts\conda env create -f %BIN_PATH%\environment.yml

rem TODO execute our PY_APP