@echo off
rem This script will remove the miniconda folder and its subfolders
rem You may need to change the paths and variables according to your project

rem Set the variables
set BIN_PATH=%~dp0
set ROOT_PATH=%BIN_PATH%..\
set MINICONDA_INSTALLER=%BIN_PATH%\miniconda_installer.exe
set MINICONDA_PATH=%BIN_PATH%\..\miniconda

rem Remove the miniconda folder and its subfolders
echo Working...
rd /s /q %MINICONDA_PATH%
echo Done!
