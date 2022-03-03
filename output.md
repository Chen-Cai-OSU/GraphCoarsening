======================================================= Execution of following cmds: =======================================================
/home/chen/anaconda3/envs/sparsifier/bin/python -W                             ignore sparsenet/model/example.py 
--lap                          none 
--train_idx                    0 
--test_idx                     0 
--n_bottomk                    40 
--force_pos 
--n_cycle                      100 
--seed                         0 
--train_indices                0 
--test_indices                 , 
--dataset                      ws 
--n_epoch                      20 
--device                       cpu 
--w_len                        5000 
--ratio                        0.7 
--method                       variation_neighborhood 
--loss                         quadratic 
--dynamic 
--ini 
--valeigen 
--train_indices                0,1,2,3, 
--test_indices                 5,6,7,8,9,10,11, 
--n_layer                      3 
--emb_dim                      50                             

/data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/sparsenet/data/nonegographs/ws/processed/ws
Finish loading dataset ws (len: 50)
train_indices: [0, 1, 3].
 val_indices: [2]. 
 test_indices: [5, 6, 7, 8, 9, 10, 11].
load g_sml, assignment, and C from 
 /data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/result/model/../coarse_graph/loukas/ws/ratio_0.7/method_variation_neighborhood/n_bottomk_40/cur_idx_0/
N: 154 x: 512 edge_attr: 5120 edge_indices: 5120
Summary of g_sml (in set loader) (torch_geometric.data.data.Data):
      edge_attr           FloatTensor         [1382]             1.95(mean)      1.0(min)     15.0(max)      1.0(median)      2.4(std)     15.0(unique)
      edge_index          LongTensor          [2, 1382]         76.25(mean)      0.0(min)    153.0(max)     78.0(median)    44.47(std)    154.0(unique)
      edge_weight         FloatTensor         [1382]             1.95(mean)      1.0(min)     15.0(max)      1.0(median)      2.4(std)     15.0(unique)
      x                   FloatTensor         [154, 5]           0.08(mean)     0.02(min)     0.17(max)     0.08(median)     0.02(std)    281.0(unique)
691 Subgraph Stats:
node_stats: array (int64) of shape   (691,)     Nan ratio:      0.0.     8.674(mean)      2.0(min)     22.0(max)     10.0(median)    6.769(std)     21.0(unique) 
edge_stats: array (int64) of shape   (691,)     Nan ratio:      0.0.    50.333(mean)      2.0(min)    180.0(max)     56.0(median)   51.116(std)     67.0(unique) 
L1                  sparse.FloatTensor  [512, 512]          0.0(mean)     -1.0(min)     13.0(max)      0.0(median)     0.47(std)      8.0(unique)
L2_baseline0        sparse.FloatTensor  [154, 154]          0.0(mean)    -15.0(min)     43.0(max)      0.0(median)     1.82(std)     42.0(unique)
Train: L1_combsparse.FloatTensor  [512, 512]          0.0(mean)     -1.0(min)     13.0(max)      0.0(median)     0.47(std)      8.0(unique)
Train: L2_ini_combsparse.FloatTensor  [154, 154]          0.0(mean)    -15.0(min)     43.0(max)      0.0(median)     1.82(std)     42.0(unique)
INFO:root:Initial quaratic loss: 4.527.
INFO:root:ws-Idx 0-Epoch: 1. Train(0.6): 1.738(0.701) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 2. Train(0.5): 1.44(0.61) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 3. Train(0.5): 1.386(0.593) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 4. Train(0.5): 1.361(0.585) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 5. Train(0.5): 1.319(0.571) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 6. Train(0.5): 1.316(0.57) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 7. Train(0.5): 1.286(0.56) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 8. Train(0.5): 1.267(0.554) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 9. Train(0.5): 1.243(0.546) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 10. Train(0.5): 1.216(0.537) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 11. Train(0.5): 1.205(0.533) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 12. Train(0.5): 1.189(0.527) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 13. Train(0.5): 1.195(0.529) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 14. Train(0.5): 1.169(0.52) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 15. Train(0.5): 1.18(0.524) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 16. Train(0.5): 1.146(0.512) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 17. Train(0.8): 1.162(0.518) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 18. Train(0.5): 1.175(0.523) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 19. Train(0.5): 1.104(0.498) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
INFO:root:ws-Idx 0-Epoch: 20. Train(0.5): 1.18(0.524) / 4.527(1.293) / 1.712. Eigenloss: -1.0. Bl_Eigenloss: 2.471
=================================================== Finish training ws 0 for 20 epochs. ====================================================

train: 10.5s
Test: Load 2-th shape from dataset ws.
load g_sml, assignment, and C from 
 /data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/result/model/../coarse_graph/loukas/ws/ratio_0.7/method_variation_neighborhood/n_bottomk_40/cur_idx_2/
N: 214 x: 712 edge_attr: 7120 edge_indices: 7120
Summary of g_sml (in set loader) (torch_geometric.data.data.Data):
      edge_attr           FloatTensor         [1892]             2.09(mean)      1.0(min)     16.0(max)      1.0(median)     2.55(std)     15.0(unique)
      edge_index          LongTensor          [2, 1892]        106.46(mean)      0.0(min)    213.0(max)    107.0(median)    62.01(std)    214.0(unique)
      edge_weight         FloatTensor         [1892]             2.09(mean)      1.0(min)     16.0(max)      1.0(median)     2.55(std)     15.0(unique)
      x                   FloatTensor         [214, 5]           0.07(mean)      0.0(min)     0.13(max)     0.07(median)     0.02(std)    357.0(unique)
946 Subgraph Stats:
node_stats: array (int64) of shape   (946,)     Nan ratio:      0.0.     8.606(mean)      2.0(min)     22.0(max)      9.0(median)    6.367(std)     21.0(unique) 
edge_stats: array (int64) of shape   (946,)     Nan ratio:      0.0.    48.275(mean)      2.0(min)    170.0(max)     43.0(median)   47.222(std)     79.0(unique) 
-------------------------------------------------------- Finish setting ws graph 2 ---------------------------------------------------------
[18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
INFO:root:            Graph-ws: 2. Test-Val(0.5):  1.48(0.768) / 3.917(1.4) / 1.3. . Eigenloss: 4.034. Bl_Eigenloss: 2.669.
===================================================== ws: finish validating graph [2]. =====================================================

-0.5112305141276349 -1e+30
Save model for train idx 0. Best-eigen-ratio is -0.51.
load g_sml, assignment, and C from 
 /data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/result/model/../coarse_graph/loukas/ws/ratio_0.7/method_variation_neighborhood/n_bottomk_40/cur_idx_1/
N: 184 x: 612 edge_attr: 6120 edge_indices: 6120
Summary of g_sml (in set loader) (torch_geometric.data.data.Data):
      edge_attr           FloatTensor         [1630]             2.04(mean)      1.0(min)     14.0(max)      1.0(median)      2.4(std)     14.0(unique)
      edge_index          LongTensor          [2, 1630]         90.58(mean)      0.0(min)    183.0(max)     89.5(median)    52.69(std)    184.0(unique)
      edge_weight         FloatTensor         [1630]             2.04(mean)      1.0(min)     14.0(max)      1.0(median)      2.4(std)     14.0(unique)
      x                   FloatTensor         [184, 5]           0.07(mean)     0.01(min)     0.16(max)     0.07(median)     0.02(std)    313.0(unique)
815 Subgraph Stats:
node_stats: array (int64) of shape   (815,)     Nan ratio:      0.0.     8.374(mean)      2.0(min)     22.0(max)      6.0(median)    6.667(std)     21.0(unique) 
edge_stats: array (int64) of shape   (815,)     Nan ratio:      0.0.    47.578(mean)      2.0(min)    156.0(max)     18.0(median)   49.755(std)     69.0(unique) 
L1                  sparse.FloatTensor  [612, 612]          0.0(mean)     -1.0(min)     14.0(max)      0.0(median)     0.43(std)     10.0(unique)
L2_baseline0        sparse.FloatTensor  [184, 184]          0.0(mean)    -14.0(min)     41.0(max)      0.0(median)     1.68(std)     47.0(unique)
Train: L1_combsparse.FloatTensor  [612, 612]          0.0(mean)     -1.0(min)     14.0(max)      0.0(median)     0.43(std)     10.0(unique)
Train: L2_ini_combsparse.FloatTensor  [184, 184]          0.0(mean)    -14.0(min)     41.0(max)      0.0(median)     1.68(std)     47.0(unique)
INFO:root:Initial quaratic loss: 4.045.
INFO:root:ws-Idx 1-Epoch: 1. Train(0.6): 0.984(0.528) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 2. Train(0.5): 0.956(0.516) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 3. Train(0.5): 0.925(0.503) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 4. Train(0.5): 0.896(0.49) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 5. Train(0.5): 0.903(0.494) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 6. Train(0.6): 0.868(0.479) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 7. Train(0.5): 0.89(0.488) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 8. Train(0.5): 0.901(0.493) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 9. Train(0.5): 0.855(0.473) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 10. Train(0.6): 0.92(0.501) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 11. Train(0.5): 0.873(0.481) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 12. Train(0.7): 0.876(0.482) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 13. Train(0.6): 0.865(0.477) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 14. Train(0.5): 0.834(0.463) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 15. Train(0.5): 0.84(0.466) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 16. Train(0.6): 0.826(0.46) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 17. Train(0.5): 0.809(0.452) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 18. Train(0.6): 0.791(0.444) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 19. Train(0.6): 0.805(0.45) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
INFO:root:ws-Idx 1-Epoch: 20. Train(0.6): 0.791(0.444) / 4.045(1.35) / 1.415. Eigenloss: -1.0. Bl_Eigenloss: 2.769
=================================================== Finish training ws 1 for 20 epochs. ====================================================

train: 11.2s
Test graph 2 has been processed. Skip.
[24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
INFO:root:            Graph-ws: 2. Test-Val(0.4):  0.583(0.375) / 3.917(1.4) / 1.3. Generalize!. Eigenloss: 2.895. Bl_Eigenloss: 2.669.
===================================================== ws: finish validating graph [2]. =====================================================

-0.08458650488901587 -0.5112305141276349
Save model for train idx 1. Best-eigen-ratio is -0.08.
load g_sml, assignment, and C from 
 /data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/result/model/../coarse_graph/loukas/ws/ratio_0.7/method_variation_neighborhood/n_bottomk_40/cur_idx_3/
N: 244 x: 812 edge_attr: 8120 edge_indices: 8120
Summary of g_sml (in set loader) (torch_geometric.data.data.Data):
      edge_attr           FloatTensor         [2110]             2.12(mean)      1.0(min)     17.0(max)      1.0(median)      2.6(std)     16.0(unique)
      edge_index          LongTensor          [2, 2110]        117.47(mean)      0.0(min)    243.0(max)    115.0(median)    70.16(std)    244.0(unique)
      edge_weight         FloatTensor         [2110]             2.12(mean)      1.0(min)     17.0(max)      1.0(median)      2.6(std)     16.0(unique)
      x                   FloatTensor         [244, 5]           0.06(mean)     0.01(min)     0.13(max)     0.06(median)     0.02(std)    374.0(unique)
1055 Subgraph Stats:
node_stats: array (int64) of shape  (1055,)     Nan ratio:      0.0.     8.287(mean)      2.0(min)     22.0(max)      7.0(median)    6.433(std)     21.0(unique) 
edge_stats: array (int64) of shape  (1055,)     Nan ratio:      0.0.    46.328(mean)      2.0(min)    172.0(max)     28.0(median)   48.142(std)     82.0(unique) 
get_subgraphs: 1.0s
L1                  sparse.FloatTensor  [812, 812]          0.0(mean)     -1.0(min)     13.0(max)      0.0(median)     0.37(std)      9.0(unique)
L2_baseline0        sparse.FloatTensor  [244, 244]          0.0(mean)    -17.0(min)     40.0(max)      0.0(median)      1.5(std)     45.0(unique)
Train: L1_combsparse.FloatTensor  [812, 812]          0.0(mean)     -1.0(min)     13.0(max)      0.0(median)     0.37(std)      9.0(unique)
Train: L2_ini_combsparse.FloatTensor  [244, 244]          0.0(mean)    -17.0(min)     40.0(max)      0.0(median)      1.5(std)     45.0(unique)
INFO:root:Initial quaratic loss: 3.371.
INFO:root:ws-Idx 3-Epoch: 1. Train(0.9): 0.612(0.441) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 2. Train(0.7): 0.589(0.427) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 3. Train(0.7): 0.578(0.421) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 4. Train(0.7): 0.552(0.405) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 5. Train(0.7): 0.556(0.408) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 6. Train(0.7): 0.542(0.399) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 7. Train(0.7): 0.535(0.395) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 8. Train(0.7): 0.529(0.391) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 9. Train(0.7): 0.539(0.397) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 10. Train(0.7): 0.527(0.39) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 11. Train(0.6): 0.53(0.392) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 12. Train(0.7): 0.519(0.385) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 13. Train(0.7): 0.512(0.38) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 14. Train(0.7): 0.515(0.383) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 15. Train(0.6): 0.5(0.373) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 16. Train(0.6): 0.495(0.37) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 17. Train(0.6): 0.506(0.377) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 18. Train(0.6): 0.513(0.381) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 19. Train(0.7): 0.493(0.369) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
INFO:root:ws-Idx 3-Epoch: 20. Train(0.7): 0.492(0.368) / 3.371(1.399) / 1.105. Eigenloss: -1.0. Bl_Eigenloss: 2.707
=================================================== Finish training ws 3 for 20 epochs. ====================================================

train: 14.1s
Test graph 2 has been processed. Skip.
[17 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
INFO:root:            Graph-ws: 2. Test-Val(0.4):  25.883(3.053) / 3.917(1.4) / 1.3. . Eigenloss: 7.042. Bl_Eigenloss: 2.669.
===================================================== ws: finish validating graph [2]. =====================================================

-1.6384826910912478 -0.08458650488901587
Test: Load 5-th shape from dataset ws.
load g_sml, assignment, and C from 
 /data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/result/model/../coarse_graph/loukas/ws/ratio_0.7/method_variation_neighborhood/n_bottomk_40/cur_idx_5/
N: 304 x: 1012 edge_attr: 10120 edge_indices: 10120
__construct_dict: 1.2s
Summary of g_sml (in set loader) (torch_geometric.data.data.Data):
      edge_attr           FloatTensor         [2654]             2.12(mean)      1.0(min)     17.0(max)      1.0(median)     2.64(std)     17.0(unique)
      edge_index          LongTensor          [2, 2654]        150.26(mean)      0.0(min)    303.0(max)    150.0(median)    87.14(std)    304.0(unique)
      edge_weight         FloatTensor         [2654]             2.12(mean)      1.0(min)     17.0(max)      1.0(median)     2.64(std)     17.0(unique)
      x                   FloatTensor         [304, 5]           0.06(mean)      0.0(min)     0.14(max)     0.06(median)     0.02(std)    484.0(unique)
1327 Subgraph Stats:
node_stats: array (int64) of shape  (1327,)     Nan ratio:      0.0.     8.711(mean)      2.0(min)     22.0(max)      9.0(median)    6.138(std)     21.0(unique) 
edge_stats: array (int64) of shape  (1327,)     Nan ratio:      0.0.    48.635(mean)      2.0(min)    170.0(max)     44.0(median)   45.716(std)     82.0(unique) 
get_subgraphs: 1.3s
-------------------------------------------------------- Finish setting ws graph 5 ---------------------------------------------------------
[20 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
INFO:root:            Graph-ws: 5. Test-Val(0.5):  0.407(0.329) / 3.408(1.449) / 1.0. Generalize!. Eigenloss: 2.707. Bl_Eigenloss: 2.51.
======================================================= ws: finish testing graph 5. ========================================================

Test: Load 6-th shape from dataset ws.
load g_sml, assignment, and C from 
 /data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/result/model/../coarse_graph/loukas/ws/ratio_0.7/method_variation_neighborhood/n_bottomk_40/cur_idx_6/
N: 334 x: 1112 edge_attr: 11120 edge_indices: 11120
__construct_dict: 1.1s
Summary of g_sml (in set loader) (torch_geometric.data.data.Data):
      edge_attr           FloatTensor         [2948]             2.11(mean)      1.0(min)     18.0(max)      1.0(median)     2.69(std)     17.0(unique)
      edge_index          LongTensor          [2, 2948]        165.91(mean)      0.0(min)    333.0(max)    166.0(median)    97.02(std)    334.0(unique)
      edge_weight         FloatTensor         [2948]             2.11(mean)      1.0(min)     18.0(max)      1.0(median)     2.69(std)     17.0(unique)
      x                   FloatTensor         [334, 5]           0.05(mean)     0.01(min)     0.11(max)     0.05(median)     0.01(std)    499.0(unique)
1474 Subgraph Stats:
node_stats: array (int64) of shape  (1474,)     Nan ratio:      0.0.     8.382(mean)      2.0(min)     22.0(max)      7.0(median)    6.209(std)     21.0(unique) 
edge_stats: array (int64) of shape  (1474,)     Nan ratio:      0.0.      46.3(mean)      2.0(min)    176.0(max)     32.0(median)   46.282(std)     82.0(unique) 
get_subgraphs: 1.4s
-------------------------------------------------------- Finish setting ws graph 6 ---------------------------------------------------------
[15 20 22 23 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
INFO:root:            Graph-ws: 6. Test-Val(0.6):  0.415(0.359) / 2.963(1.407) / 1.0. Generalize!. Eigenloss: 2.782. Bl_Eigenloss: 2.472.
======================================================= ws: finish testing graph 6. ========================================================

Test: Load 7-th shape from dataset ws.
load g_sml, assignment, and C from 
 /data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/result/model/../coarse_graph/loukas/ws/ratio_0.7/method_variation_neighborhood/n_bottomk_40/cur_idx_7/
N: 364 x: 1212 edge_attr: 12120 edge_indices: 12120
__construct_dict: 1.2s
Summary of g_sml (in set loader) (torch_geometric.data.data.Data):
      edge_attr           FloatTensor         [3196]             2.13(mean)      1.0(min)     17.0(max)      1.0(median)      2.6(std)     16.0(unique)
      edge_index          LongTensor          [2, 3196]        180.18(mean)      0.0(min)    363.0(max)    180.0(median)    104.8(std)    364.0(unique)
      edge_weight         FloatTensor         [3196]             2.13(mean)      1.0(min)     17.0(max)      1.0(median)      2.6(std)     16.0(unique)
      x                   FloatTensor         [364, 5]           0.05(mean)     0.01(min)     0.15(max)     0.05(median)     0.01(std)    549.0(unique)
1598 Subgraph Stats:
node_stats: array (int64) of shape  (1598,)     Nan ratio:      0.0.     8.379(mean)      2.0(min)     22.0(max)      8.0(median)    6.065(std)     21.0(unique) 
edge_stats: array (int64) of shape  (1598,)     Nan ratio:      0.0.    45.687(mean)      2.0(min)    168.0(max)     34.0(median)   44.745(std)     82.0(unique) 
get_subgraphs: 1.5s
-------------------------------------------------------- Finish setting ws graph 7 ---------------------------------------------------------
[17 24 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
INFO:root:            Graph-ws: 7. Test-Val(0.6):  0.368(0.331) / 2.939(1.418) / 0.9. Generalize!. Eigenloss: 2.76. Bl_Eigenloss: 2.517.
======================================================= ws: finish testing graph 7. ========================================================

Test: Load 8-th shape from dataset ws.
load g_sml, assignment, and C from 
 /data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/result/model/../coarse_graph/loukas/ws/ratio_0.7/method_variation_neighborhood/n_bottomk_40/cur_idx_8/
N: 394 x: 1312 edge_attr: 13120 edge_indices: 13120
__construct_dict: 1.3s
Summary of g_sml (in set loader) (torch_geometric.data.data.Data):
      edge_attr           FloatTensor         [3418]             2.21(mean)      1.0(min)     20.0(max)      1.0(median)     2.79(std)     18.0(unique)
      edge_index          LongTensor          [2, 3418]         195.6(mean)      0.0(min)    393.0(max)    194.0(median)   113.42(std)    394.0(unique)
      edge_weight         FloatTensor         [3418]             2.21(mean)      1.0(min)     20.0(max)      1.0(median)     2.79(std)     18.0(unique)
      x                   FloatTensor         [394, 5]           0.05(mean)     0.01(min)     0.11(max)     0.05(median)     0.01(std)    554.0(unique)
1709 Subgraph Stats:
node_stats: array (int64) of shape  (1709,)     Nan ratio:      0.0.     8.324(mean)      2.0(min)     22.0(max)      7.0(median)    5.915(std)     21.0(unique) 
edge_stats: array (int64) of shape  (1709,)     Nan ratio:      0.0.    44.735(mean)      2.0(min)    158.0(max)     28.0(median)   44.007(std)     79.0(unique) 
get_subgraphs: 1.8s
-------------------------------------------------------- Finish setting ws graph 8 ---------------------------------------------------------
[22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
INFO:root:            Graph-ws: 8. Test-Val(0.6):  0.37(0.341) / 2.837(1.414) / 0.9. Generalize!. Eigenloss: 2.731. Bl_Eigenloss: 2.361.
======================================================= ws: finish testing graph 8. ========================================================

Test: Load 9-th shape from dataset ws.
load g_sml, assignment, and C from 
 /data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/result/model/../coarse_graph/loukas/ws/ratio_0.7/method_variation_neighborhood/n_bottomk_40/cur_idx_9/
N: 424 x: 1412 edge_attr: 14120 edge_indices: 14120
__construct_dict: 1.7s
Summary of g_sml (in set loader) (torch_geometric.data.data.Data):
      edge_attr           FloatTensor         [3812]             2.12(mean)      1.0(min)     19.0(max)      1.0(median)     2.74(std)     17.0(unique)
      edge_index          LongTensor          [2, 3812]        210.07(mean)      0.0(min)    423.0(max)    211.0(median)   122.28(std)    424.0(unique)
      edge_weight         FloatTensor         [3812]             2.12(mean)      1.0(min)     19.0(max)      1.0(median)     2.74(std)     17.0(unique)
      x                   FloatTensor         [424, 5]           0.05(mean)     0.01(min)     0.13(max)     0.05(median)     0.01(std)    622.0(unique)
1906 Subgraph Stats:
node_stats: array (int64) of shape  (1906,)     Nan ratio:      0.0.     8.422(mean)      2.0(min)     22.0(max)      7.0(median)    6.082(std)     21.0(unique) 
edge_stats: array (int64) of shape  (1906,)     Nan ratio:      0.0.    45.357(mean)      2.0(min)    176.0(max)     28.0(median)   44.721(std)     82.0(unique) 
get_subgraphs: 1.8s
-------------------------------------------------------- Finish setting ws graph 9 ---------------------------------------------------------
[24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
INFO:root:            Graph-ws: 9. Test-Val(0.7):  0.32(0.303) / 2.841(1.422) / 0.9. Generalize!. Eigenloss: 2.536. Bl_Eigenloss: 2.343.
======================================================= ws: finish testing graph 9. ========================================================

Test: Load 10-th shape from dataset ws.
load g_sml, assignment, and C from 
 /data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/result/model/../coarse_graph/loukas/ws/ratio_0.7/method_variation_neighborhood/n_bottomk_40/cur_idx_10/
N: 454 x: 1512 edge_attr: 15120 edge_indices: 15120
__construct_dict: 1.5s
Summary of g_sml (in set loader) (torch_geometric.data.data.Data):
      edge_attr           FloatTensor         [3888]             2.23(mean)      1.0(min)     18.0(max)      1.0(median)     2.78(std)     18.0(unique)
      edge_index          LongTensor          [2, 3888]        226.35(mean)      0.0(min)    453.0(max)    225.0(median)   129.92(std)    454.0(unique)
      edge_weight         FloatTensor         [3888]             2.23(mean)      1.0(min)     18.0(max)      1.0(median)     2.78(std)     18.0(unique)
      x                   FloatTensor         [454, 5]           0.04(mean)     0.01(min)     0.11(max)     0.05(median)     0.01(std)    605.0(unique)
1944 Subgraph Stats:
node_stats: array (int64) of shape  (1944,)     Nan ratio:      0.0.     8.072(mean)      2.0(min)     22.0(max)      6.0(median)     5.98(std)     21.0(unique) 
edge_stats: array (int64) of shape  (1944,)     Nan ratio:      0.0.    43.346(mean)      2.0(min)    176.0(max)     20.0(median)   44.714(std)     82.0(unique) 
get_subgraphs: 1.9s
-------------------------------------------------------- Finish setting ws graph 10 --------------------------------------------------------
[18 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
INFO:root:            Graph-ws: 10. Test-Val(0.7):  0.292(0.311) / 2.441(1.397) / 0.8. Generalize!. Eigenloss: 2.705. Bl_Eigenloss: 2.452.
======================================================= ws: finish testing graph 10. =======================================================

Test: Load 11-th shape from dataset ws.
load g_sml, assignment, and C from 
 /data/chen/cai.507/cai.507/Documents/DeepLearning/GraphCoarsening/result/model/../coarse_graph/loukas/ws/ratio_0.7/method_variation_neighborhood/n_bottomk_40/cur_idx_11/
N: 484 x: 1612 edge_attr: 16120 edge_indices: 16120
__construct_dict: 1.9s
Summary of g_sml (in set loader) (torch_geometric.data.data.Data):
      edge_attr           FloatTensor         [4242]             2.16(mean)      1.0(min)     18.0(max)      1.0(median)     2.82(std)     17.0(unique)
      edge_index          LongTensor          [2, 4242]        241.46(mean)      0.0(min)    483.0(max)    243.0(median)   140.09(std)    484.0(unique)
      edge_weight         FloatTensor         [4242]             2.16(mean)      1.0(min)     18.0(max)      1.0(median)     2.82(std)     17.0(unique)
      x                   FloatTensor         [484, 5]           0.04(mean)     0.01(min)     0.11(max)     0.04(median)     0.01(std)    646.0(unique)
2121 Subgraph Stats:
node_stats: array (int64) of shape  (2121,)     Nan ratio:      0.0.     8.048(mean)      2.0(min)     22.0(max)      6.0(median)    6.026(std)     21.0(unique) 
edge_stats: array (int64) of shape  (2121,)     Nan ratio:      0.0.    43.087(mean)      2.0(min)    178.0(max)     22.0(median)   44.579(std)     82.0(unique) 
get_subgraphs: 2.0s
-------------------------------------------------------- Finish setting ws graph 11 --------------------------------------------------------
[13 17 21 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
INFO:root:            Graph-ws: 11. Test-Val(0.8):  0.299(0.319) / 2.464(1.409) / 0.8. Generalize!. Eigenloss: 2.618. Bl_Eigenloss: 2.369.
======================================================= ws: finish testing graph 11. =======================================================

/home/chen/anaconda3/envs/sparsifier/bin/python -W                             ignore sparsenet/model/example.py 
--lap                          none 
--train_idx                    0 
--test_idx                     0 
--n_bottomk                    40 
--force_pos 
--n_cycle                      100 
--seed                         0 
--train_indices                0 
--test_indices                 , 
--dataset                      ws 
--n_epoch                      20 
--device                       cpu 
--w_len                        5000 
--ratio                        0.7 
--method                       variation_neighborhood 
--loss                         quadratic 
--dynamic 
--ini 
--valeigen 
--train_indices                0,1,2,3, 
--test_indices                 5,6,7,8,9,10,11, 
--n_layer                      3 
--emb_dim                      50                             

