import yaml, numpy as np, copy 
from pprint import pprint 


default_config = {
  "avg_pool_kernel_size": 4, 
  "conv_kernel_sizes": [3, 3, 3, 3],
  "num_blocks": [2, 2, 2, 2] ,
  "num_channels": 64,
  "shortcut_kernel_sizes": [1, 1, 1, 1] ,
  "drop": 0, 
  "squeeze_and_excitation": 0, 
  "max_epochs": 200,
  "optim": "sgd" ,
  "lr_sched": "CosineAnnealingLR",
  "momentum": 0.9,
  "lr": 0.1 ,
  "weight_decay": 0.0005 ,
  "batch_size": 128,
  "num_workers": 16,
  "resume_ckpt": 0,
  "data_augmentation": 1, 
  "data_normalize": 1, 
  "grad_clip": 0.1 
} 
config = {} 
for name in ["ResNet18", "baseline_ResNet"]: 
    for dropout in np.arange(0.1, 0.8, 0.05): 
        dropout = round(float(dropout), 2)  
        exp = name + "_dropout_" + str(dropout) 
        config[exp] = copy.deepcopy(default_config)
        config[exp]['drop'] = dropout 

with open('resnet_configs/dropoutResNets.yaml', 'w') as file:
    yaml.dump(config, file) 



