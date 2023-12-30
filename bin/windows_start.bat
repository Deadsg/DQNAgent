@echo off
rem This script will install miniconda locally to the game.exe of the build
rem a: K-Rawson y:2024

rem Set the variables
set BIN_PATH=%~dp0
set ROOT_PATH=%BIN_PATH%..\
set DQNA_PATH=%ROOT_PATH%\DQNAgent
set MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
set MINICONDA_INSTALLER=%BIN_PATH%\miniconda_installer.exe
set MINICONDA_PATH=%ROOT_PATH%\miniconda
set REQUIREMENTS_TXT=requirements.txt
set REQUIREMENTS_PATH=%ROOT_PATH%\%REQUIREMENTS_TXT%

rem TODO this should be app.py calling DQNAgent.py
set APP_PY=DQNAgent.py
set APP_PATH=%DQNA_PATH%\%APP_PY%

rem Download the miniconda installer
bitsadmin /transfer miniconda_download %MINICONDA_URL% %MINICONDA_INSTALLER%

echo Install Miniconda...
rem Install miniconda silently to the local folder
%MINICONDA_INSTALLER% /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_PATH%

rem Delete the miniconda installer
del %MINICONDA_INSTALLER%

rem Run python command
set PY_COMMAND=pip install -r %REQUIREMENTS_PATH%
%MINICONDA_PATH%\%PY_COMMAND%
echo Done!