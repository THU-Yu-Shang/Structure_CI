# Structure-CI
Implementation of running Social group network, Resnet and ER/BA/WS network 

Results(**num of paths and performance**) are saved in **results.xlsx**

### Data preparation
* Download [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) then put the folder to the **randwire/data, resnet/data, socialnet/data** 
### Social group network
* Run training of Socialgroup network 
```
cd socialnet
python main3.py --alpha 1 --modularity 4 --gs 0
```
* When exploring unevenness of speaking turns, set **modularity=4**, change giniA by **alpha**; change giniB by **gs**(generally set as 0).
* When exploring modularity, set **alpha=1** and **gs=1**, change num of subgroup by **modularity** 
* Note: In Socialgroup network, num of paths can be calculated by multiplication of adjacency matrix of different rounds.
### Resnet
* Run training of Resnet
```
cd resnet
python train.py -net skip_resnet34
```
* change **self.distance_p** in **skip_resnet.py** to get different wired Resnet structure

### ER/BA/WS network
* Run training of Randwire network 
```
cd randwire
python main.py --graph_mode BA --m 1
python main.py --graph_mode WS --k 4 --p 0.5
python main.py --graph_mode ER --p 0.5
```

### Calculate num of paths from the adjacent matrix in WS/BA/ER network
Due to the time complexity of searching all paths in large graphs, we implement the algorithm mentioned in [《Estimating the number of s-t paths in a graph》](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.143.6025&rep=rep1&type=pdf) to estimate the number of paths for the given graph(WS/ER/BA graph).
* Copy the content of **adj_matrix.txt** produced by training of models above and replace the matrix **A** in **social/R/social.all.paths.R**
* Run **social.all.paths.R** to get the estimated number of paths 
