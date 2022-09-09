import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision import models
import numpy as np
#from models.resnet import resnet34
#model = resnet34()
from models.prun_resnet import resnet34
model = resnet34()
model.load_state_dict(torch.load('/nfs/shangy/pytorch-cifar100/checkpoint/prun_resnet34/Thursday_21_July_2022_00h_40m_57s/prun_resnet34-187-best.pth'))
d_counter = []
for name,param in model.named_parameters():
        #if 'skip' in name and 'weight' in name:
            print('name:',name)
            #print(param.data.numpy())
            #d_counter.append(param.data.numpy())

# cum_degree = np.cumsum(sorted(np.append(d_counter, 0)))
# sum_degree = cum_degree[-1]
# xarray = np.array(range(0, len(cum_degree))) / np.float(len(cum_degree) - 1)
# yarray = cum_degree / sum_degree
# B = np.trapz(yarray, x=xarray)
# A = 0.5 - B
# G = A / (A + B)
# print('G:',G)
# train.py
#!/usr/bin/env	python3

# """ train network using pytorch

# author baiyu
# """

# import os
# import sys
# import argparse
# import time
# from datetime import datetime

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms

# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# from conf import settings
# from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
#     most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

# @torch.no_grad()
# def eval_training(epoch=0, tb=True):

#     start = time.time()
#     net.eval()

#     test_loss = 0.0 # cost function error
#     correct = 0.0

#     for (images, labels) in cifar100_test_loader:

#         if args.gpu:
#             images = images.cuda()
#             labels = labels.cuda()

#         outputs = net(images)
#         loss = loss_function(outputs, labels)

#         test_loss += loss.item()
#         _, preds = outputs.max(1)
#         correct += preds.eq(labels).sum()

#     finish = time.time()
#     #if args.gpu:
#      #   print('GPU INFO.....')
#         #print(torch.cuda.memory_summary(), end='')
#     print('Evaluating Network.....')
#     print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
#         epoch,
#         test_loss / len(cifar100_test_loader.dataset),
#         correct.float() / len(cifar100_test_loader.dataset),
#         finish - start
#     ))
#     print()

#     #add informations to tensorboard
#     if tb:
#         writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
#         writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

#     return correct.float() / len(cifar100_test_loader.dataset)

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-net', type=str, required=True, help='net type')
#     parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
#     parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
#     parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
#     parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
#     parser.add_argument('-resume', action='store_true', default=True, help='resume training')
#     args = parser.parse_args()

#     net = get_network(args)

#     #data preprocessing:
#     cifar100_training_loader = get_training_dataloader(
#         settings.CIFAR100_TRAIN_MEAN,
#         settings.CIFAR100_TRAIN_STD,
#         num_workers=4,
#         batch_size=args.b,
#         shuffle=True
#     )

#     cifar100_test_loader = get_test_dataloader(
#         settings.CIFAR100_TRAIN_MEAN,
#         settings.CIFAR100_TRAIN_STD,
#         num_workers=4,
#         batch_size=args.b,
#         shuffle=True
#     )

#     loss_function = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#     train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
#     iter_per_epoch = len(cifar100_training_loader)
#     warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

#     if args.resume:
#         recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
#         if not recent_folder:
#             raise Exception('no recent folder were found')

#         checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

#     else:
#         checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

#     #use tensorboard
#     if not os.path.exists(settings.LOG_DIR):
#         os.mkdir(settings.LOG_DIR)

#     #since tensorboard can't overwrite old values
#     #so the only way is to create a new tensorboard log
#     writer = SummaryWriter(log_dir=os.path.join(
#             settings.LOG_DIR, args.net, settings.TIME_NOW))
#     input_tensor = torch.Tensor(1, 3, 32, 32)
#     if args.gpu:
#         input_tensor = input_tensor.cuda()
#     writer.add_graph(net, input_tensor)

#     #create checkpoint folder to save model
#     if not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path)
#     checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
#     #checkpoint_path = '/nfs/shangy/pytorch-cifar100/checkpoint/skip_resnet34/Tuesday_28_June_2022_16h_52m_42s/skip_resnet34-200-best.pth'
#     best_acc = 0.0
#     if args.resume:
#             a = torch.load('/nfs/shangy/pytorch-cifar100/checkpoint/skip_resnet34/Tuesday_28_June_2022_16h_52m_42s/skip_resnet34-200-best.pth')
#             for k in a.keys():
#                 print(k)
#             #best_acc = eval_training(tb=False)
#             #print('best acc is {:0.2f}'.format(best_acc))

#     writer.close()
