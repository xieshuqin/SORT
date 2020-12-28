#SORT
A PyTorch implementation of [SORT](https://arxiv.org/pdf/1602.00763) algorithm.
## Installation
### Requirements
```
pip install motmetrics
```
### Dataset
Place the `2DMOT2015` dataset under folder `./data`. 

## Instruction
```
python main.py
python -m motmetrics.apps.eval_motchallenge ./data/train ./results
```