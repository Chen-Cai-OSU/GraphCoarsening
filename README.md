# sparsifier

## Install

* python 3.7.6
* torch                 1.4.0               
* pytorch geometric  1.4.3 --> 1.5.0
* networkx              2.4

## Example

 python sparsenet/model/model.py  (results may differ but shape should be same. see TODO.)
 
```
edges                    LongTensor          [2, 100]       152.56(mean)   0.0(min) 314.0(max) 147.5(median) 86.45(std)  83.0(unique)
----------
g.x                      FloatTensor         [320, 42]        0.5(mean)   0.0(min)   1.0(max)   0.5(median)  0.29(std) 13438.0(unique)
g.edge_index             LongTensor          [2, 100]       152.56(mean)   0.0(min) 314.0(max) 147.5(median) 86.45(std)  83.0(unique)
g.edge_attr              LongTensor          [100, 20]        0.0(mean)   0.0(min)   0.0(max)   0.0(median)   0.0(std)   1.0(unique)
pred                     FloatTensor         [8, 18]         -0.0(mean) -0.05(min)  0.06(max)  0.01(median)  0.03(std)  18.0(unique)
pred                     FloatTensor         [8, 18]         -0.0(mean) -0.05(min)  0.06(max)  0.01(median)  0.03(std)  18.0(unique)
pred                     FloatTensor         [8, 18]         -0.0(mean) -0.05(min)  0.06(max)  0.01(median)  0.03(std)  18.0(unique)
pred                     FloatTensor         [8, 18]         -0.0(mean) -0.05(min)  0.06(max)  0.01(median)  0.03(std)  18.0(unique)
```

## TODO
* solve the random seed issue to make the results reproducible (Done)
* Implement loss (Done)
* Effective resistance in networkx is slow. Look for faster options 
* Implement Dan Speilman's algorithm. Maybe can find it in [here](https://github.com/TheGravLab/A-Unifying-Framework-for-Spectrum-Preserving-Graph-Sparsification-and-Coarsening).
* Get familar with sanity_check.py. It will be useful later on.



## Question
* Right now, we have two models. One GNN to get node feature for sparsified graph G'. One Edge weighter to assign
the edge weights for G'. Can we merge two models? Done.
* The batch for two models. One Batch for GNN corresponds to a few subgraphs.
One batch for edge weighter corresponds to a few edges for G'
How to balance batch of GNN and batch of edge weigher.Done


