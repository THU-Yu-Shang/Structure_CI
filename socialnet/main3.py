import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import inspect 
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
#from torchstat import stat
import argparse
import os
import networkx as nx
import time
from tqdm import tqdm
import random
from model3 import Model
from preprocess import load_data
from plot import draw_plot
import torch
from gpu_mem_track import MemTracker

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(torch.cuda.is_available())

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def adjust_learning_rate(optimizer, epoch, args):
    #lr = args.learning_rate * (0.1 ** (epoch // 30))
    lr = args.learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    for data, target in tqdm(train_loader, desc="epoch " + str(epoch), mininterval=1):
        adjust_learning_rate(optimizer, epoch, args)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output,_,_,_,_ = model(data)
        #break
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        #train_loss += loss.item()
        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        if step % 100 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))

    length = len(train_loader.dataset) // args.batch_size
    return train_loss / length, train_acc / length


def get_test(model, test_loader,save):
    model.eval()
    correct = 0
    num_batch = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="evaluation", mininterval=1):
            #break
            data, target = data.to(device), target.to(device)
            output,sd,g,e,memory = model(data)
            # if save:
            #     if num_batch<78:
            #         np.save('sim/out_tensor2/agent.npy', memory)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()
            num_batch = num_batch + 1

    acc = 100. * float(correct) / len(test_loader.dataset)
    return acc,sd,g,e


def main():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=201, help='number of epochs, (default: 100)')
    parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.75)')
    parser.add_argument('--c', type=int, default=154, help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    parser.add_argument('--k', type=int, default=7, help='each node is connected to k nearest neighbors in ring topology, (default: 4)')
    parser.add_argument('--m', type=int, default=3, help='number of edges to attach from a new node to existing nodes, (default: 5)')
    parser.add_argument('--graph-mode', type=str, default="BA", help="random graph, (Example: ER, WS, BA), (default: WS)")
    parser.add_argument('--node-num', type=int, default=12, help="Number of graph node (default n=32)")
    parser.add_argument('--learning-rate', type=float, default=0.1, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size, (default: 100)')
    parser.add_argument('--model-mode', type=str, default="SMALL_REGIME", help='CIFAR10, CIFAR100, SMALL_REGIME, REGULAR_REGIME, (default: CIFAR10)')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR100", help='Which dataset to use? (Example, CIFAR10, CIFAR100, MNIST), (default: CIFAR10)')
    parser.add_argument('--is-train', type=bool, default=True, help="True if training, False if test. (default: True)")
    parser.add_argument('--load-model', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=1,help='The expotential term in preference attachment, change giniA')
    parser.add_argument('--modularity', type=float, default=4, help='Subgroup number')
    parser.add_argument('--gs', type=float, default=2, help='Group shuffle level, change giniB')


    args = parser.parse_args()

    train_loader, test_loader = load_data(args)

    if args.load_model:
            model = Model(args.node_num, args.modularity, args.gs, args.alpha, args.p, args.k, args.m, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train).to(device)
            filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode
            checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
            model.load_state_dict(checkpoint['model'])
            epoch = checkpoint['epoch']
            acc = checkpoint['acc']
            print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)
    else:
            model = Model(args.node_num, args.modularity, args.gs, args.alpha, args.p, args.k, args.m, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train).to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    # for name, module in model.named_modules():
    #     print(name)
    #     print(module)
    #total = sum([param.nelement() for param in model.parameters()])
    #print("Number of parameters: %.2fM" % (total/1e6))
    
    epoch_list = []
    test_acc_list = []
    train_acc_list = []
    train_loss_list = []
    max_test_acc = 0
    
    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    start_time = time.time()
    with open("./reporting/" + "m_" +str(args.m) + "_lr_" +str(args.learning_rate) + "_channel_" +str(args.c) +".txt", "w") as f:

        for epoch in range(1, args.epochs + 1):

        #for epoch in range(1):
            # scheduler = CosineAnnealingLR(optimizer, epoch)
            epoch_list.append(epoch)
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, args)
            #gpu_tracker.track() 
            if epoch <200:
                test_acc,sd,g,e = get_test(model, test_loader,False)
                #gpu_tracker.track() 
            else:
                test_acc,sd,g,e = get_test(model, test_loader,True)
            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            print('Test set accuracy: {0:.2f}%, Best accuracy: {1:.2f}%'.format(test_acc, max_test_acc))
            f.write("[Epoch {0:3d}]  Train loss:{1:.3f}%, Train acc:{2:.3f}%, Test set accuracy: {3:.3f}%,  Best accuracy: {4:.2f}%".format(epoch, train_loss, train_acc, test_acc, max_test_acc))
            f.write("Weighted: SD: {0:.8f}, G: {1:.8f} ,E: {2:.8f}".format(sd, g, e))
            print('write done')
            f.write("\n ")

            if max_test_acc < test_acc:
                print('Saving..')
                state = {
                    'model': model.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                filename = "c_" + str(args.c) + "_alpha_" + str(
                    args.alpha) + "_" + args.graph_mode
                #torch.save(state, './checkpoint/' + filename + 'ckpt.t7')
                #torch.save(model.state_dict(), 'case_model/'+filename+'.pth')
                max_test_acc = test_acc
                #draw_plot(epoch_list, train_loss_list, train_acc_list, test_acc_list)
            print("Training time: ", time.time() - start_time)
            f.write("Training time: " + str(time.time() - start_time))
            f.write("\n")
            #gpu_tracker.track() 
            

if __name__ == '__main__':
    main()
