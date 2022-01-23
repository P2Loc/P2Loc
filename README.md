# P2Loc
This repository is the official PyTorch implementation of the *P2Loc* reported in the paper: <br>
[*P2-Loc: A Person-2-Person Indoor Localization System in On-Demand Delivery*](). 

## Installation
Requirements: Python >= 3.5, [Anaconda3](https://www.anaconda.com/)

- Update conda:
```bash
conda update -n base -c defaults conda
```

- Install basic dependencies to virtual environment and activate it: 
```bash
conda env create -f environment.yml
conda activate degnn-env
```

- Install PyTorch >= 1.4.0 and torch-geometric >= 1.5.0
```bash
conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
```

The latest tested combination is: Python 3.8.2 + Pytorch 1.4.0 + torch-geometric 1.5.0.

## Quick Start
```
python main.py
```
