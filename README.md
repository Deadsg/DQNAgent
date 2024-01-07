# DQN Agent
This project is a work in progress that aims to create a deep Q-network (DQN) agent system that can learn from its own experiences and adapt to different environments. The system uses a self-learning Q loop algorithm, which is a variant of Q-learning that updates the Q-values based on the agent’s actions and rewards. The system also uses LLm learning, which is a technique that leverages the latent space of the neural network to improve the generalization and robustness of the agent. The project is written in Python and uses PyTorch as the framework for the neural network.

# Description

This project aims to develop a deep Q-network (DQN) agent system that can learn from its own experiences and adapt to different environments. DQN is a reinforcement learning method that uses a neural network to approximate the optimal action-value function. The system will use a self-learning Q loop algorithm, which is a variant of Q-learning that updates the Q-values based on the agent’s actions and rewards. The system will also use LLm learning, which is a technique that leverages the latent space of the neural network to improve the generalization and robustness of the agent.
 
# Install

## Windows

First you need to clone this repository. Using your preferred terminal console to navigate where you cloned this repository. Then run the following command in the terminal

```console
> .\bin\windows_setup.bat
```

This will download and install a local miniconda python environment. It will then install the required dependendencies.

## Linux

 - Coming Soon!
 
## Mac

 - Coming Soon!
 
# Usage
 
After you have installed the system with `windows_setup.bat` next you need to run `windows_run.bat`

```console
> .\bin\windows_run.bat
```

This will start the DQN Agent in your windows console terminal.

# Development

The project uses Visual Studio (or VS Code if preferred) and requires the project to be setup before loading into visual studio.

## Setup Local Development (Windows)
To setup for local development perform the following steps

 1. Git clone the repository
 2. open your command terminal (or powershell)
 3. cd into where you cloned the repository
 4. run `.\bin\windows_setup.bat` 
 5. wait for setup to finish
 6. Open the `DQNAgent.sln`
 7. wait for solution explorer to load
 8. In Visual Studio, Open `tools -> python -> Python Environments`
 9. In `Python Environments` window click on `Add Environment`
 10. Select `Existing Environment` 
 11. In the environment dropdown select `<Custom>`
 12. In `Prefix Path` browse to `<repo_location>\DQNAgent\miniconda\envs\DQNAgent`
 13. Click `Add`
 14. Select `DQNAgent` from dropdown in visual studio toolbar