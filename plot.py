from tensorboard.backend.event_processing.event_accumulator import EventAccumulator 
import matplotlib.pyplot as plt, glob, pprint, numpy as np 

def moving_average(a, n=20) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
for exp in glob.glob('plot_summaries/conv_kernels/*'): 
    file = glob.glob(exp+'*/event*')[0] 
    print(file)

    event_acc = EventAccumulator(file)
    event_acc.Reload()
    _, _, val1 = zip(*event_acc.Scalars('Accuracy/test_accuracy'))
    axs[0].plot(moving_average(np.array(val1[:125])), label=exp.split('/')[-1])  
    axs[0].legend() 
    axs[0].set_title("Accuracy/test_accuracy")
    # axs[0].x_label("Accuracy")
    # axs[0].y_label("Epochs")
    _, _, val1 = zip(*event_acc.Scalars('Loss/test_loss'))
    axs[1].plot(moving_average(np.array(val1[:125])), label=exp.split('/')[-1])  
    axs[1].legend() 
    axs[1].set_title("Loss/test_loss")

plt.show() 
plt.close() 

# plt.savefig('plots/conv_kernels.png') 