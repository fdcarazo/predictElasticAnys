import matplotlib.pyplot as plt
## plot loss-.
class PlotLosses():
    def __init__(self):
        pass

    def plot_loss(self,label_str:str,val:list):
        fig, ax= plt.subplots(nrows=1,ncols=1,figsize=(12,6))
        plt.plot(range(len(val)), val, 'ro', label=label_str)
        plt.grid(color='b',linestyle='--',linewidth='0.5')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.legend(loc='best')
        ax.set_title('training loss using Bayesian Neural Network')
        plt.show()
