import yaml, numpy as np, copy 
from pprint import pprint 


ResNet18_default_config = {
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

Vanilla_default_config = {
  "avg_pool_kernel_size": 4, 
  "conv_kernel_sizes": [3, 3, 3, 3],
  "num_blocks": [2, 2, 2, 2] ,
  "num_channels": 64,
  "shortcut_kernel_sizes": [1, 1, 1, 1],
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
  "data_augmentation": 0, 
  "data_normalize": 0, 
  "grad_clip": 0 
} 

config = {} 
for name in ["vanilla_ResNet4"]: 
  for num_blocks in [[2,1,1,1], [1,1,1,1]]: 
    for num_channels in [16, 32, 64]: 
      # for conv in [3, 5, 7]: 
          exp = name 
          exp += "_num_blocks" + str(num_blocks)
          exp += "_num_channels" + str(num_channels) 
          # exp += "_conv" + str(conv) 
          config[exp] = copy.deepcopy(Vanilla_default_config)
          config[exp]['num_blocks'] = copy.deepcopy(num_blocks) 
          config[exp]['num_channels'] = num_channels  
          # config[exp]['conv'] = [conv]*len(num_blocks) 
print(len(config.keys()))
pprint(config.keys())
# exit() 


"""
for name in ["ResNet18", "baseline_ResNet"]: 
  for dadn in [0, 1]: 
    for drop in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]: 
      for grad_clip in [0, 0.1]: 
          exp = name 
          if dadn: exp += "_dadn" + str(dadn)
          if drop: exp += "_drop" + str(drop) 
          if grad_clip: exp += "_gradclip" + str(grad_clip) 
          config[exp] = copy.deepcopy(ResNet18_default_config)
          if name == "baseline_ResNet": 
              config[exp]['num_blocks'] = [2,1,1,1] 
          config[exp]['data_augmentation'] = dadn 
          config[exp]['data_normalize'] = dadn  
          config[exp]['drop'] = drop 
          config[exp]['grad_clip'] = grad_clip 
# print(len(config.keys()))
# pprint(config.keys())
# exit() 
"""
"""
for name in ["ResNet18", "baseline_ResNet"]: 
  exp = name + "_se" 
  config[exp] = copy.deepcopy(default_config)
  if name == "baseline_ResNet": 
      config[exp]['num_blocks'] = [2,1,1,1] 
  config[exp]['squeeze_and_excitation'] = 1  
"""

# # Dropout 
# for name in ["ResNet18", "baseline_ResNet"]: 
#   for dropout in np.arange(0.1,0.8,0.05): 
#     dropout = round(float(dropout), 2)
#     exp = name + "_dropout_" + str(dropout)
#     config[exp] = copy.deepcopy(default_config)
#     if name == "baseline_ResNet": 
#         config[exp]['num_blocks'] = [2,1,1,1] 
#     config[exp]['drop'] = dropout 

with open('resnet_configs/sunday_vanilla_ResNets4.yaml', 'w') as file:
    yaml.dump(config, file) 



