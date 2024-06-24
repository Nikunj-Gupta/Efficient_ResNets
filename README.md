# Mini-Project 1: Residual Network Design 
*Authors:* Nikunj Gupta, Harish Chauhan, Aditya Thakur 

**Summary:** We constructed a Residual Network Design with less than 5 million trainable parameters. We achieved an accuracy of 96.04% on CIFAR-10 dataset by using best-suited hyperparameters and multiple training strategies like data normalization, data augmentation, optimizers, gradient clipping, etc.

# Introduction 
ResNets (or Residual Networks) are one of the most commonly used models for image classification
5 tasks. In this project, you will design and train your own ResNet model for CIFAR-10 image
6 classification. In particular, your goal will be to maximize accuracy on the CIFAR-10 benchmark
7 while keeping the size of your ResNet model under budget. Model size, typically measured as the
8 number or trainable parameters, is important when models need to be stored on devices with limited
9 storage capacity, mobile devices for example. 

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

To install all the dependencies, execute: `pip install -r requirements.txt`

# Description of files in the repository 
- models/resnet.py : PyTorch description of ResNet model architecture (flexible to change/modify using config.yaml) 
- main.py : code to train and test ResNet architectures 
- config.yaml : contains the hyperparamters used for constructing and training a ResNet architecture 
- project1_model.pt : Trained parameters/weights for our final model.
- project1_model.py : ResNet architecture used.

# Training
```
# Start training with: 
python3 main.py  --config <path_to_config> --resnet_architecture <architecture_id>
```
To modify and test with new ResNet architectures, you can create a new configuration experiment in `config.yaml`. Currently, it includes descriptions for our model and ResNet18. 

# Reproduce the results 

#### Train our best modified ResNet Architecture with: 
```
python3 main.py  --config resnet_configs/config.yaml --resnet_architecture best_model
```
We have set the above as our default inputs in `main.py` and hence the following will reproduce our results too: 
```
python3 main.py 
```

#### Train ResNet18 Architecture with: 
```
python3 main.py  --config resnet_configs/config.yaml --resnet_architecture resnet18
```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| Our ResNet          | 96.04%      |
| ResNet18           | 88.56%     |

# Hyperparameters in our final model's architecture 

| Parameter                    | Our Model       |
| ---------------------------- | --------------- |
|number of residual layers     |3                |
|number of residual blocks | [4, 4, 3]| 
|convolutional kernel sizes |[3, 3, 3] |
|shortcut kernel sizes |[1, 1, 1] |
|number of channels |64 |
|average pool kernel size |8|
|batch normalization |True |
|dropout |0 |
|squeeze and excitation |True|
|gradient clip |0.1|
|data augmentation |True|
|data normalization |True|
|lookahead |True |
|optimizer |SGD|
|learning rate (lr)| 0.1|
|lr scheduler |CosineAnnealingLR|
|weight decay |0.0005|
|batch size |128 |
|number of workers |16|
|Total number of Parameters| 4,697,742|

## Cite 
```
@article{thakur2023efficient,
  title={Efficient ResNets: Residual Network Design},
  author={Thakur, Aditya and Chauhan, Harish and Gupta, Nikunj},
  journal={arXiv preprint arXiv:2306.12100},
  year={2023}
}
```
