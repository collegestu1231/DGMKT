# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np
import torch
import scipy.sparse as sp


def generate_G_from_H(H, variable_weight=False):
    # 这个函数 generate_G_from_H 的主要功能是从超图的关联矩阵(incidence matrix)
    # H 生成图的结构矩阵 G，它用于表示节点之间的连接关系，通常在超图卷积神经网络中用作信息传播的工具。
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]  # Number of columns of matrix = number of hyperedge 边 204=102*2
    #print(H.shape)(6324, 204)
    # the weight of the hyperedge
    W = np.ones(n_edge)  # 每个边初始化权重为1
    # print(W.shape) (204,)
    # the degree of the node
    DV = np.sum(H * W, axis=1)  # 节点的度
    # print(DV.shape)(6324,) 一共6324个节点的度
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # 一共204条边的度
    # print(DE.shape)(204,)
    invDE = np.mat(np.diag(np.power(DE, float(-1))))  # 超边度的倒数形成的对角矩阵
    #print(invDE.shape)(204, 204)
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))  # 节点度的平方根倒数形成的对角矩阵，表示节点的影响力。
    #print(DV2.shape)(6324, 6324)
    W = np.mat(np.diag(W))
    #print(W.shape)(204, 204)
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        # print("DV2",DV2)
        # print("H",H)
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        # print("DV2", DV2)
        # print("H", H)
        # DV2*H=A对原矩阵的行(节点)*乘以对应的度的倒数，减少度大的点的贡献
        # A*W = B 对A中的边进行一个权重分配
        # B*invDE = C 对B中的边进行归一化
        # C*HT = D 将超边的影响回传给节点。这一步生成了初步的节点间连接矩阵，它反映了超边如何将信息从一个节点传播到另一个节点。
        G = DV2 * H * W * invDE * HT * DV2


        G = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(G))
        # G = torch.Tensor(G)
        return G


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.把一个sparse matrix转为torch中的稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
