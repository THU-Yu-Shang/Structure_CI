import torch
from torchvision import datasets, transforms

import argparse
import os
from tqdm import tqdm
from torchvision.models import densenet161
from model import Model
from preprocess import load_data

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def main():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, (default: 100)')
    parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.75)')
    parser.add_argument('--c', type=int, default=144, help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    parser.add_argument('--k', type=int, default=7, help='each node is connected to k nearest neighbors in ring topology, (default: 4)')
    parser.add_argument('--m', type=int, default=8, help='number of edges to attach from a new node to existing nodes, (default: 5)')
    parser.add_argument('--graph-mode', type=str, default="BA", help="random graph, (Example: ER, WS, BA), (default: WS)")
    parser.add_argument('--node-num', type=int, default=32, help="Number of graph node (default n=32)")
    parser.add_argument('--learning-rate', type=float, default=1e-1, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size, (default: 100)')
    parser.add_argument('--model-mode', type=str, default="SMALL_REGIME", help='CIFAR10, CIFAR100, SMALL_REGIME, REGULAR_REGIME, (default: CIFAR10)')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR100", help='Which dataset to use? (Example, CIFAR10, CIFAR100, MNIST), (default: CIFAR10)')
    parser.add_argument('--is-train', type=bool, default=True, help="True if training, False if test. (default: True)")
    parser.add_argument('--load-model', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=1e-20)

    args = parser.parse_args()

    _, test_loader = load_data(args)
    '''
    if os.path.exists("./checkpoint"):
        model = Model(args.node_num, 1, args.p, args.k, args.m, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train).to(device)
        filename = "c_" + str(args.c) + "_p_" + str(args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode
        checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
        model.load_state_dict(checkpoint['model'])
        end_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
        print("[Saved Best Accuracy]: ", best_acc, '%', "[End epochs]: ", end_epoch)

        model.eval()
        correct = 0
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            y_pred = output.data.max(1)[1]
            correct += y_pred.eq(target.data).sum()
        print("[Test Accuracy] ", 100. * float(correct) / len(test_loader.dataset), '%')

    else:
        assert os.path.exists("./checkpoint/" + str(args.seed) + "ckpt.t7"), "File not found. Please check again."
    '''
    #model = Model(args.node_num, 1, args.p, args.k, args.m, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train).to(device)
    model = densenet161()
    print("Number of model parameters: ", get_model_parameters(model))


if __name__ == "__main__":
    main()
