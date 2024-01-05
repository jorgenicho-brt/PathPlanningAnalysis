# PathPlanningAlgorithms
Path Planning Algorithms with C++ Wrappers and Python Bindings 

## First Time Setup 
Currently some artifacts must be downloaded from artifactory to build this project correctly. You must setup a jfrog access token in order to authenticate with artifactory:
1. Follow instructions to obtain a developer token: https://developerexperience.deere.com/tools/j/jfrog/access/#identity-tokens
1. Once you obtain your token, store it by adding `export JFROG_API_TOKEN="YOUR_API_TOKEN"` in your `~/.bashrc`
1. Run `./setupEnv.sh` to install cmake, ninja, jfrog, protobuf compiler
1. Run `./download_ppl.sh` to download Path Planning Library headers and libraries and copy them to the appropriate project folders 
1. Initialize git submodules

## Building C++ project:
1. Run `./build_cpp.sh`

A file called `PyPL.so` will be built and located in `/build/lib`. This file is the importable python module that contains interfaces for each algorithm

## Jupyter Setup
1. Install miniconda https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
2. Create conda-jupyter environment
```
cd jupyter
conda env create -f environment.yml
```
3. Activate Conda path planning environment
```
conda activate path-planner
```
4. Run Jupyter lab
```
jupyter lab
```

if the default port is unavailable then run
```
jupyter lab --port 8080
```
> Change the port number if needed

## Simulation 
WIP
