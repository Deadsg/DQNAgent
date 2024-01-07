@echo off
rem This script will remove the miniconda folder and its subfolders
rem You may need to change the paths and variables according to your project

rem Set the variables
set BIN_PATH=%~dp0
set ROOT_PATH=%BIN_PATH%..\
set MINICONDA_INSTALLER=%BIN_PATH%\miniconda_installer.exe
set MINICONDA_PATH=%BIN_PATH%\..\miniconda
set MODELS_ELEUTHERAI_GPT_NEOX_20B=models--EleutherAI--gpt-neox-20b
set HUGGINGFACE_PATH=%USERPROFILE%\.cache\huggingface\hub

rem Remove the miniconda folder and its subfolders
echo Working...
rd /s /q %MINICONDA_PATH%
rd /s /q %HUGGINGFACE_PATH%\%MODELS_ELEUTHERAI_GPT_NEOX_20B%
rd /s /q %HUGGINGFACE_PATH%\.locks\%MODELS_ELEUTHERAI_GPT_NEOX_20B%
del /s /f /q "%HUGGINGFACE_PATH%\tmp*."
echo Done!
