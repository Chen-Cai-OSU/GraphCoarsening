# Created at 2020-09-02
# Summary: a toy example to check laplacian

import numpy as np

L = np.array([[3, -1, -1, -1, 0],
              [-1, 3, -1, 0, -1],
              [-1, -1, 2, 0, 0],
              [-1, 0, 0, 1, 0],
              [0, -1, 0, 0, 1]
              ])

L1 = np.array([[3, -1, -1, -1, 0],
              [-1, 3, -1, 0, -1],
              [-1, -1, 2, 0, 0],
              [-1, 0, 0, 1, 0],
              [0, -1, 0, 0, 1]
              ])

L2 = np.array([[5, -2, -1, -2, 0],
              [-2, 6, -3, 0, -1],
              [-1, -3, 4, 0, 0],
              [-2, 0, 0, 2, 0],
              [0, -1, 0, 0, 1]
              ])

P = np.array([[1 / 3, 1 / 3, 1 / 3, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]])

C = np.sqrt(P)

Pinv = np.array([[1, 0, 0],
                 [1, 0, 0],
                 [1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])

L_c = Pinv.T @ L @ Pinv

def normlap(L):
    D_ = np.diag(1.0 / np.sqrt(np.diag(L))) #
    return D_ @ L @ D_

def proj(L):
    return Pinv.T @ L @ Pinv

if __name__ == '__main__':
    print(proj(L1 + L2))
    exit()
    print('L_c', L_c)
    print('L_c_norm', normlap(L_c))
    L_c_norm = normlap(L_c)

    print('L', L)
    print('L_norm', normlap(L))
    L_norm = normlap(L)


    ret = L_norm
    print(ret)
    print(np.multiply(ret, ret))