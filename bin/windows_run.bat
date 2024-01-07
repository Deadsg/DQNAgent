@echo off
rem This script will activate the virtual environment and run the dqnagent.py script

set BIN_PATH=%~dp0
set ROOT_PATH=%BIN_PATH%..\
set DQNA_TEXT=DQNAgent
set OKAY_TEXT=Okay!
set DQNA_PATH=%ROOT_PATH%\%DQNA_TEXT%
set MINICONDA_PATH=%ROOT_PATH%\miniconda

echo.

:initialize_conda_env
echo Initialize Conda Environment
call %MINICONDA_PATH%\Scripts\conda init --quiet powershell
echo %OKAY_TEXT%
echo.

:activate_conda_env
echo Activate Conda Environment
call %MINICONDA_PATH%\condabin\activate %DQNA_TEXT%
echo %OKAY_TEXT%
echo.

:run_dqn_agent
echo Run DQN Agent
python %DQNA_PATH%\DQNAgent.py
echo %OKAY_TEXT%
echo.

:deactivate_conda_env
echo Deactivate Conda Environment
conda deactivate
echo %OKAY_TEXT%
echo.