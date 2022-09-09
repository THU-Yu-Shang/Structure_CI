import matplotlib.pyplot as plt
import os
import numpy as np

def draw_plot(epoch_list, train_loss_list, train_acc_list, val_acc_list, a_list):
    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    #print(type(epoch_list))
    #print(type(train_loss_list))
    plt.plot(epoch_list, train_loss_list, label='training loss')
    plt.legend()

    plt.subplot(132)
    plt.plot(epoch_list, train_acc_list, label='train acc')
    plt.plot(epoch_list, val_acc_list, label='validation acc')
    plt.legend()

    plt.subplot(133)
    plt.plot(list(np.arange(len(a_list))), a_list, label='alpha')
    plt.legend()

    if os.path.isdir('./plot1'):
        plt.savefig('./plot1/epoch_acc_plot.png')

    else:
        os.makedirs('./plot1')
        plt.savefig('./plot1/epoch_acc_plot.png')
    plt.close()
