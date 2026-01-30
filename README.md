# Template Python

This is the description of my project!

This template hold all the base tools for a python project according to our standards.
It has the ruff linter, for formating and linting the code.
It has the capability to build self-contained executable on windows and linux and generate a manifest file to publish it
to dist.

Folder structure:  
.  
├── .github                 # Github templates   
├── .vscode                 # Vscode settings  
├── ci                      # All files used in the jenkins file    
├── tests                   # package tests  
├── template                # package  
├── external                # submodule  
├── Jenkinsfile             # pipeline definition  
├── version.txt             # file containing only package version  as plain text   
├── whitelabel.txt          # file containing only package whitelabel as plain text   
└── pyproject.toml          # package configuration  

# Development setup

## Intellij:

Install ruff plugin:

- go to : settings > plugin
- search ruff and install plugin
- go to settings > tools > ruff
- check enable formating

## vscode

Install recommended plugins. Vscode will propose to install the plugins since .vscode/extensions contains the full list.

Ruff will be set as default formatter with the proper configuration.

## Install dev package

Install the package in development will all dependencies

```shell
python3 -m pip install -e .[all]
```

In Ubuntu 24.04:
```shell
python3 -m pip install -e .[all] --break-system-packages
```

## Creating a venv

create .venv

```shell
./scripts/with_dev_venv.sh python3 -m pip install -e .[all]
```

activate venv

```shell 
source .venv/bin/activate
```

Deactivate venv

```shell
deactivate
```

# Build whl

A whl package can be build with the following script

```shell
./ci/build_package.sh
```


## dependencies: 

https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/6ef/f643c0a7acda9/torch-2.7.0-cp310-cp310-linux_aarch64.whl#sha256=6eff643c0a7acda92734cc798338f733ff35c7df1a4434576f5ff7c66fc97319
https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/daa/bff3a07259968/torchvision-0.22.0-cp310-cp310-linux_aarch64.whl#sha256=daabff3a0725996886b92e4b5dd143f5750ef4b181b5c7d01371a9185e8f0402
https://pypi.jetson-ai-lab.dev/jp6/cu126
numpy<2