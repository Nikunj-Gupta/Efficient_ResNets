from tensorboard.backend.event_processing.event_accumulator import EventAccumulator 
import matplotlib.pyplot as plt, glob, pprint, numpy as np 

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

for exp in glob.glob('./summaries/ResNet18*'): 
    file = glob.glob(exp+'*/event*')[0]
    print(exp.split('/'[-1]))
    event_acc = EventAccumulator(file)
    event_acc.Reload()
    # print(event_acc.Tags()) 
    w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy/test_accuracy')) 
    # w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy/train_accuracy')) 
    plt.plot(moving_average(np.array(vals[:-1])), label=exp.split('/')[-1])  
plt.legend() 
plt.show() 