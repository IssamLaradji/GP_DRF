# Efficient Deep Gaussian Process Models for Variable-Sized Inputs - IJCNN 2019  
[[Paper](https://arxiv.org/abs/1905.06982)]

## Overview
Our proposed method combines Gaussian Processes with deep random feature expansion. This repository combines Gaussian processes (GP), deep random feature (DRF) model, and our GP-DRF model.

## Requirements

- Pytorch version 0.4 or higher.

## Running the methods
You can run each example as follows.

- For Gaussian processes,
```
python GP_example.py
```

- For the deep random feature expansion model,
```
python DRF_example.py
```

- For our GP-DRF model,
```
python GP_DRF_example.py
```

## Citation
If you find this useful, please consider citing us!

```
@article{laradji2019efficient,
  title={Efficient Deep Gaussian Process Models for Variable-Sized Input},
  author={Laradji, Issam H and Schmidt, Mark and Pavlovic, Vladimir and Kim, Minyoung},
  journal={arXiv preprint arXiv:1905.06982},
  year={2019}
}
```
