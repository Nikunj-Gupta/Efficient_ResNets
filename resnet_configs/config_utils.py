import yaml 
from pprint import pprint 


config = {} 
count = 1 
for num_blocks in [[1,1,1,1], [2,1,1,1]]: 
    for num_channels in range(32, 32*3, 32):         
        for avg_pool_kernel_size in range(1, 4): 
            for conv_kernel_size in range(2, 4): 
                for shortcut_kernel_sizes in range(2, 4): 

                    name = "nikResNet_"+str(count) 
                    config[name] = {} 

                    config[name]['num_blocks'] = list(num_blocks)                                                           # N: number of Residual Layers | Bi:Residual blocks in Residual Layer i 
                    config[name]['conv_kernel_sizes'] = [conv_kernel_size]*len(config[name]['num_blocks'])                  # Fi: Conv. kernel size in Residual Layer i 
                    config[name]['shortcut_kernel_sizes'] = [shortcut_kernel_sizes]*len(config[name]['num_blocks'])         # Ki: Skip connection kernel size in Residual Layer i 
                    config[name]['num_channels'] = num_channels                                                             # Ci: # channels in Residual Layer i 
                    config[name]['avg_pool_kernel_size'] =  avg_pool_kernel_size                                            # P: Average pool kernel size 

                    count += 1 

print("Number of experiments: ", len(config.keys()))

with open('resnet_configs/nikResNets.yaml', 'w') as file:
    yaml.dump(config, file)




