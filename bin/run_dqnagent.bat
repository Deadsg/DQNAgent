@echo off
rem This script will activate the virtual environment and run the dqnagent.py script

set BIN_PATH=%~dp0
set ROOT_PATH=%BIN_PATH%..\
set DQNA_TEXT=DQNAgent
set MINICONDA_PATH=%ROOT_PATH%\miniconda
%MINICONDA_PATH%\Scripts\conda init
%MINICONDA_PATH%\condabin\conda.bat activate
%ROOT_PATH%\%DQNA_TEXT%\dqnagent.py
