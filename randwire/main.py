import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torchmetrics import Metric, BootStrapper
import argparse
import os
import seaborn as sns
import time
from tqdm import tqdm
import numpy as np
from model import Model
from preprocess import load_data
#from plot import draw_plot
#from einops import rearrange, repeat
#from loguru import logger
import matplotlib.pyplot as plt
plt.switch_backend('agg')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

def get_model_gradient(model):
    grads = OrderedDict()
    for name, params in model.named_parameters():
        grad = params.grad
        if grad is not None and 'conv1' in name:
            grads[name] = grad.view(-1)
    return grads

def gram(X):
    X = rearrange(X, 'b ... -> b (...)')
    return X @ X.T

def centering_mat(n):
    v_i = torch.ones(n,1, device=DEVICE)
    H = torch.eye(n, device=DEVICE) - (v_i @ v_i.T) / n
    return H

def centered_gram(X):
    K = gram(X)
    m = K.shape[0]
    H = centering_mat(m)
    #logger.info(H.shape)
    #logger.info(K.shape)
    return H @ K @ H

def unbiased_hsic_xy(X,Y):
    n = X.shape[0]
    assert n > 3 
    v_i = torch.ones(n,1, device=DEVICE)

    K = gram(X)
    diag_K = torch.diag(K)
    K_diag = torch.diag_embed(diag_K)
    K_hat = K - K_diag

    L = gram(Y)
    diag_L = torch.diag(L)
    L_diag = torch.diag_embed(diag_L)
    L_hat = L - L_diag

    KL = K_hat @ L_hat
    iK = v_i.T @ K_hat
    Li = L_hat @ v_i
    iKi = iK @ v_i
    iLi = v_i.T @ Li

    a = torch.trace(KL)
    b = iKi * iLi / ((n-1)*(n-2))
    c = iK @ Li * 2 / (n-2)

    outv = (a + b - c) / (n*(n-3))

    # if a+b-c == 0:
    #     print(a)
    #     print(b)
    #     print(c)
    #print(outv)
    return outv.item()

class MinibatchCKA(Metric):
    def __init__(self, dist_sync_on_step=False):

        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("_xx", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("_xy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("_yy", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, X: torch.Tensor, Y: torch.Tensor):
        self._xx += unbiased_hsic_xy(X,X)
        self._xy += unbiased_hsic_xy(X,Y)
        self._yy += unbiased_hsic_xy(Y,Y)
        # if unbiased_hsic_xy(X,X) == 0:
        #     print(X)
        #     print(Y)
        #print(Y)
    def compute(self):
        xx, xy, yy = self._xx, self._xy, self._yy
        result = xy / (torch.sqrt(xx) * torch.sqrt(yy))
        if torch.isnan(result):
            result = torch.tensor(0.0)
        return result

class HookedCache:
    def __init__(self, model, target):
        self.model = model
        self.target = target
        
        self.clear()
        self._extract_target()
        self._register_hook()

    @property
    def value(self):
        return self._cache
    def clear(self):
        self._cache = None
    def _extract_target(self):
        #print(self.target)
        for name, module in self.model.named_modules():
          #print(name)
          if name == self.target:
              print('get!')
              self._target = module
              return
    def _register_hook(self):
        def _hook(module, in_val, out_val):
             self._cache = out_val
        self._target.register_forward_hook(_hook)

def get_simmat_from_metrics(metrics):
    vals = []
    for i, ckas in enumerate(metrics):
      for j, cka in enumerate(ckas):
        z = cka.compute().item()
        vals.append((i,j,z))

    sim_mat = torch.zeros(i+1,j+1)
    for i,j,z in vals:
      sim_mat[i,j] = z

    delete_index = []
    for d in range(sim_mat.size()[0]):
        temp = sim_mat[d,:][sim_mat[d,:]>0]
        temp = temp[temp<=1]
        if len(temp)==0:
            delete_index.append(d)

    # for b in range(sim_mat.size()[0]-1):
    #     for c in range(b+1,sim_mat.size()[0]):
    #         if abs(sim_mat[b,c]-sim_mat[c,b])>0.2:
    #             sim_mat[b,c] = 1
    #             sim_mat[c,b] = 1 
    #         if sim_mat[b,c] < 0:
    #             sim_mat[b,c] = 0 
    #             sim_mat[c,b] = 0    
    #         if sim_mat[b,c] > 1:
    #             sim_mat[b,c] = 1
    #             sim_mat[c,b] = 1 

    # for r in range(sim_mat.size()[0]):
    #     sim_mat[r,r] = 1
    
    #print(torch.any(torch.isnan(sim_mat)))
    # sur_sum = 0
    # count = 0
    # for p in range(sim_mat.size()[0]-1):
    #     for q in range(p+1,sim_mat.size()[0]):
    #         if p in delete_index:
    #             count = count+1
    #         else:
    #             sur_sum += 1-sim_mat[p,q]
    #             count = count+1

    #si = 0.5*(torch.sum(sim_mat.view(-1))-sim_mat.size()[0])
    #si = si*2/(sim_mat.size()[0]*(sim_mat.size()[0]-1))
    #si = sur_sum/count
    #print('dead agent:',delete_index)
    #print('SI:',si)
    return sim_mat

def make_pairwise_metrics(mod1_hooks, mod2_hooks):
  metrics = []
  for i_ in mod1_hooks:
    metrics.append([])
    for j_ in mod2_hooks:
      metrics[-1].append(MinibatchCKA().to(DEVICE))
  return metrics

def update_metrics(mod1_hooks, mod2_hooks, metrics, metric_name, do_log):
    for i, hook1 in enumerate(mod1_hooks):
      for j, hook2 in enumerate(mod2_hooks):
        cka = metrics[i][j]
        X,Y = hook1.value, hook2.value
        X = X+1e-4
        Y = Y+1e-4
        #print(torch.count_nonzero(Y.view(-1)).item())

        cka.update(X,Y)
        if do_log and 0 in (i,j):
          _metric_name = f"{metric_name}_{i}-{j}"
          v = cka.compute()
          #writer.add_scalar(_metric_name, v, it)
    if do_log:
       sim_mat = get_simmat_from_metrics(metrics)
       sim_mat = sim_mat.unsqueeze(0) * 255
       #writer.add_image(metric_name, sim_mat, it)

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    grad = []
    for data, target in tqdm(train_loader, desc="epoch " + str(epoch), mininterval=1):
        adjust_learning_rate(optimizer, epoch, args)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        grad_dic = get_model_gradient(model)
        temp_d = []
        for i,j in grad_dic.items():
            temp_d.append(j.cpu())
        temp = torch.cat(temp_d)
        grad.append(temp.unsqueeze(0))
        #print(len(temp))
        optimizer.step()
        train_loss += loss.data
        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        if step % 100 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))

    #calculate coherence alpha-batchnorm
    fenzi = 0
    fenmu = 0
    grad = torch.cat(grad,dim=0)
    fenzi = torch.sum(torch.mm(grad,grad.T).view(-1))
    fenmu = torch.sum(torch.mul(grad,grad).view(-1))
    #print(grad.size())
    fenzi = fenzi/(len(grad)*len(grad))
    fenmu = fenmu/(len(grad))
    alpha_batch = fenzi/fenmu
    #alpha-batch  ---->  alpha-per_example
    alpha = alpha_batch/(args.batch_size-(args.batch_size-1)*alpha_batch)
    #print('alpha:',alpha*len(grad)*args.batch_size)

    length = len(train_loader.dataset) // args.batch_size
    return (train_loss / length), (train_acc / length), alpha


def get_test(model, test_loader,save):
    model.eval()
    correct = 0
    num_batch = 0

    log_every = 10

    # modc_hooks = []
    # for i in range(5,6):
    #     for j in range(1,33):
    #         tgt = f'SMALL_conv{i}.0.module_list.{j}.unit'
    #         hook = HookedCache(model, tgt)
    #         modc_hooks.append(hook)
        
    # metrics_cc = make_pairwise_metrics(modc_hooks, modc_hooks)
    # it = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="evaluation", mininterval=1):
            # do_log =  (it % log_every == 0)
            # if do_log:
            #     logger.debug(f"iter: {it}")

            data, target = data.to(device), target.to(device)
            output = model(data)
            # if save:
            #     update_metrics(modc_hooks, modc_hooks, metrics_cc, "cka/cc", do_log)
            #     for hook0 in modc_hooks:
            #         hook0.clear()

            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()
            num_batch = num_batch + 1
            
            #it = it+1

    acc = 100. * float(correct) / len(test_loader.dataset)
    # if save:
    #     sim_mat = get_simmat_from_metrics(metrics_cc)
    #     #plt.imshow(sim_mat)
    #     #plt.title('Randwire CKA')
    #     print(sim_mat)
    #     #np.savetxt('sim_mod1_'+str(acc)+'_'+str(sur)+'.txt', sim_mat)
    #     f, ax = plt.subplots(figsize=(10, 8))
    #     ax = sns.heatmap(sim_mat, ax=ax, fmt='.3f', cmap='coolwarm',square=True)
    #     #q = ax.get_figure()
    #     #q.savefig('WS6_'+str(acc)+'.png')


    return acc


def main():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=200, help='number of epochs, (default: 100)')
    parser.add_argument('--p', type=float, default=1, help='graph probability, (default: 0.75)')
    parser.add_argument('--c', type=int, default=109, help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    parser.add_argument('--k', type=int, default=8, help='each node is connected to k nearest neighbors in ring topology, (default: 4)')
    parser.add_argument('--m', type=int, default=14, help='number of edges to attach from a new node to existing nodes, (default: 5)')
    parser.add_argument('--graph-mode', type=str, default="WS", help="random graph, (Example: ER, WS, BA), (default: WS)")
    parser.add_argument('--node-num', type=int, default=32, help="Number of graph node (default n=32)")
    parser.add_argument('--learning-rate', type=float, default=1e-1, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size, (default: 100)')
    parser.add_argument('--model-mode', type=str, default="SMALL_REGIME", help='CIFAR10, CIFAR100, SMALL_REGIME, REGULAR_REGIME, (default: CIFAR10)')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR100", help='Which dataset to use? (Example, CIFAR10, CIFAR100, MNIST), (default: CIFAR10)')
    parser.add_argument('--is-train', type=bool, default=True, help="True if training, False if test. (default: True)")
    parser.add_argument('--load-model', type=bool, default=False)

    args = parser.parse_args()

    train_loader, test_loader = load_data(args)

    if args.load_model:
        model = Model(args.node_num, args.p, args.k, args.m, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train).to(device)
        filename = "C" + str(args.c) + "_p_" + str(args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode
        checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)

        for name, module in model.named_modules():
            print(name)
            print(module)
    else:
        model = Model(args.node_num, args.p, args.k, args.m, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train).to(device)

    if device is 'cuda':
        model = torch.nn.DataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    epoch_list = []
    test_acc_list = []
    train_acc_list = []
    train_loss_list = []
    max_test_acc = 0
    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    start_time = time.time()
    with open("./reporting/" + "C_" + str(args.c) + "_p_" + str(args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode + ".txt", "w") as f:
        a_list = []
        for epoch in range(1, args.epochs + 1):
            # scheduler = CosineAnnealingLR(optimizer, epoch)
            epoch_list.append(epoch)
            train_loss, train_acc, alpha = train(model, train_loader, optimizer, criterion, epoch, args)
            a_list.append(alpha)
            if epoch <300:
                test_acc = get_test(model, test_loader,False)
            else:
                test_acc = get_test(model, test_loader,True)
            #print(train_loss)
            test_acc_list.append(torch.tensor(test_acc).cpu())
            train_loss_list.append(torch.tensor(train_loss).cpu())
            train_acc_list.append(torch.tensor(train_acc).cpu())
            print('Test set accuracy: {0:.2f}%, Best accuracy: {1:.2f}%'.format(test_acc, max_test_acc))
            f.write("[Epoch {0:3d}] Test set accuracy: {1:.3f}%, , Best accuracy: {2:.2f}%".format(epoch, test_acc, max_test_acc))
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
                filename = "C" + str(args.c) + "_p_" + str(
                    args.p) + "_graph_mode_" + args.graph_mode + "_dataset_" + args.dataset_mode
                torch.save(state, './checkpoint/' + filename + 'ckpt.t7')
                max_test_acc = test_acc
            #draw_plot(epoch_list, train_loss_list, train_acc_list, test_acc_list, a_list)
            #np.save('alpha'+"_p_" + str(args.p) + "_graph_mode_" + args.graph_mode+'.npy',np.array(a_list))
                #np.save('train_loss_list.npy',train_loss_list)
                #np.save('train_acc_list.npy',train_acc_list)
                #np.save('test_acc_list.npy',test_acc_list)
            print("Training time: ", time.time() - start_time)
            f.write("Training time: " + str(time.time() - start_time))
            f.write("\n")


if __name__ == '__main__':
    main()
