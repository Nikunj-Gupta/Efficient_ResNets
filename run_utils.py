import os 
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

for opt in ['sgd', 'adam']: 
    if opt=='sgd': 

        for lr_sched in ['CosineAnnealingLR','LambdaLR', 'MultiplicativeLR',  
                            'StepLR',  'MultiStepLR',  'ExponentialLR', 
                             'CyclicLR',  'CyclicLR2',  'CyclicLR3', 
                             'OneCycleLR',  'OneCycleLR2',  'CosineAnnealingWarmRestarts']: 
            exp = os.path.join('optimizers', '_'.join([opt, lr_sched])) 
            command = fixed_text + "#SBATCH --job-name="+exp+"\n" + "#SBATCH --output="+exp+".out\n"
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
            for i in count(0):
                temp = log_dir/('run{}.sh'.format(i)) 
                if temp.exists():
                    pass
                else:
                    with open(temp, "w") as f:
                        f.write(command) 
                    log_dir = temp
                    break
    elif opt=='adam': 
        exp = os.path.join('optimizers', opt) 
        command = fixed_text + "#SBATCH --job-name="+exp+"\n" + "#SBATCH --output="+exp+".out\n"
        command += "\nmodule load python/intel/3.8.6\n"\
                    "module load openmpi/intel/4.0.5\n"\
                    "\nsource ../venvs/dl/bin/activate\n"\
                    "time python3 main.py " 
        
        command = ' '.join([
            command, 
            "--exp", exp, 
            "--opt", opt, 
            '--data_augmentation', '--data_normalize', "--grad_clip", str(0.1), 
        ])
        # print(command) 
        log_dir = Path(dumpdir)
        for i in count(0):
            temp = log_dir/('run{}.sh'.format(i)) 
            if temp.exists():
                pass
            else:
                with open(temp, "w") as f:
                    f.write(command) 
                log_dir = temp
                break 