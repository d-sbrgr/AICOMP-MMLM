# AICOMP-MMLM

Solving the Kaggle competition "March Machine Learning Madness 2025"

---

## Setup

This project uses `python3.12` and `uv` as package management tool.

### Installing chocolatey

````shell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
````

### Installing uv

````shell
choco install uv
````

### Installing project requirements

````shell
uv sync
````

### Adding dependencies

````shell
uv add DEPENDENCY-NAME
````

### Removing dependencies

````shell
uv remove DEPENDENCY-NAME
````



