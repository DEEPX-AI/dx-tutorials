# Introduction
Welcome! This repository contains a growing collection of hands-on Jupyter Lab tutorials designed for learners of all levels who starts learning DEEPX products. Whether you're just getting started or looking to master advanced features, you'll find something useful here.

The goal is to provide clear, step-by-step guides to help you become more productive and efficient in your data science and development workflows.

## Table of Contents
* **Tutorial-01 (Getting Started)**: Introduces how to install the DeepX SDK and verify that the setup is successful.
* **Tutorial-02 (DX-APP)**: Introduce DX-APP and how to use run demos with image/video/camera inputs.
* **Tutorial-03 (E2E AI workflow)**: Hands-on practice to implement a Forklift-Worker detector with YOLOv7.
* **Tutorial-04 (DX-STREAM)**: Learn DX-STREAM and Hands-on practice with a Forklift-Worker detector + DX_STREAM.
* **Tutorial-05 (DX-Compile)**: Learn how to use DX-Compile and practice with various AI models with different preprocessing options.
* ... (and more to come!)


# Installation Guide

## Prerequisites:
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-venv build-essential python3-dev git-all ffmpeg tree
```

## Download this dx-tutorials repo
```bash
git clone https://github.com/DEEPX-AI/dx-tutorials.git
cd dx-tutorials
```


## Create/Activate a python3 virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install required pip packages:
```bash
pip install -U pip
pip install -r requirements.txt
```


# Usages
## Run jupyter-lab
```bash
./run-jupyter-lab.sh
```

## External connection (Optional)
```bash
jupyter-lab --generate-config
vi ~/.jupyter/jupyter_lab_config.py
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.open_browser = False
```
