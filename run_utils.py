import os, numpy as np 
from pathlib import Path 
from itertools import count

dumpdir = "runs/" 
if not os.path.isdir(dumpdir):
    os.mkdir(dumpdir)
fixed_text = "#!/bin/bash\n"\
             "#SBATCH --nodes=1\n"\
             "#SBATCH --cpus-per-task=16 \n"\
             "#SBATCH --time=1:00:00\n"\
             "#SBATCH --mem=40GB\n"\
             "#SBATCH --gres=gpu:1\n"


# batch size + lr 

for batch_size in range(128, 1024+256, 256): 
    for lr in np.arange(0.1,0.5,0.1): 
        exp = os.path.join('batch_size_lr', '_'.join(['batch_size'+str(batch_size), 'lr'+str(lr)])) 
        command = fixed_text + "#SBATCH --job-name="+'_'.join(['batch_size'+str(batch_size), 'lr'+str(lr)])+"\n" + "#SBATCH --output="+'_'.join(['batch_size'+str(batch_size), 'lr'+str(lr)])+".out\n"
        command += "\nmodule load python/intel/3.8.6\n"\
                    "module load openmpi/intel/4.0.5\n"\
                    "\nsource ../venvs/dl/bin/activate\n"\
                    "time python3 main.py " 
        
        command = ' '.join([
            command, 
            "--exp", exp, 
            '--batch_size', str(batch_size), 
            '--lr', str(lr), 
            '--data_augmentation', '--data_normalize', "--grad_clip", str(0.1), 
        ])

        # print(command) 
        log_dir = Path(dumpdir)
        for i in count(1):
            temp = log_dir/('run{}.sh'.format(i)) 
            if temp.exists():
                pass
            else:
                with open(temp, "w") as f:
                    f.write(command) 
                log_dir = temp
                break 





# Optimizers 
"""
for opt in ['sgd','adam']: 
    for lr_sched in ['CosineAnnealingLR','LambdaLR', 'MultiplicativeLR',  
                        'StepLR',  'MultiStepLR',  'ExponentialLR', 
                        'CyclicLR',  'CyclicLR2',  'CyclicLR3', 
                        'OneCycleLR',  'OneCycleLR2',  'CosineAnnealingWarmRestarts']: 
        exp = os.path.join('optimizers', '_'.join([opt, lr_sched])) 
        command = fixed_text + "#SBATCH --job-name="+'_'.join([opt, lr_sched])+"\n" + "#SBATCH --output="+'_'.join([opt, lr_sched])+".out\n"
        command += "\nmodule load python/intel/3.8.6\n"\
                    "module load openmpi/intel/4.0.5\n"\
                    "\nsource ../venvs/dl/bin/activate\n"\
                    "time python3 main.py " 
        
        command = ' '.join([
            command, 
            "--exp", exp, 
            "--opt", opt, 
            "--lr_sched", lr_sched, 
            '--data_augmentation', '--data_normalize', "--grad_clip", str(0.1), 
        ])

        # print(command) 
        log_dir = Path(dumpdir)
        for i in count(1):
            temp = log_dir/('run{}.sh'.format(i)) 
            if temp.exists():
                pass
            else:
                with open(temp, "w") as f:
                    f.write(command) 
                log_dir = temp
                break 
"""