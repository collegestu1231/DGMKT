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

    n_edge = H.shape[1]  # Number of columns of matrix = number of hyperedge

    # the weight of the hyperedge
    W = np.ones(n_edge)  # 每个边初始化权重为1

    DV = np.sum(H * W, axis=1)  # 节点的度
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # 边的度
    invDE = np.mat(np.diag(np.power(DE, float(-1))))  # 超边度的倒数形成的对角矩阵

    DV2 = np.mat(np.diag(np.power(DV, -0.5)))  # 节点度的平方根倒数形成的对角矩阵，表示节点的影响力。

    W = np.mat(np.diag(W))

    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:

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
