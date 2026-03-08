# Cross-Guided Attention for Robust Small Object Detection in Remote Sensing Images

# Overall architecture
<img width="1573" height="748" alt="image" src="https://github.com/user-attachments/assets/0d4733b0-c9a5-43e7-81dd-3f3b003f7505" />

# Introduction
This repository currently releases the source code of CGANet, together with all relevant training results.
Training and evaluation are performed on a workstation equipped with two NVIDIA RTX 5080 GPUs, running Ubuntu 20.04 LTS, with Python 3.9 and PyTorch 2.7.1.
epochs:200
batch:8
workers=4
optimizer='SGD'
initial learning rate:0.01
momentum:0.937
weight decay:0.0005
Box Loss Weight:7.5
Classification Loss Weight:1.5
