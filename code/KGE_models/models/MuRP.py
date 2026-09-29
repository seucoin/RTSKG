"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os
from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import *
# https://github.com/tensorflow/neural-structured-learning/blob/master/research/kg_hyp_emb/models/hyperbolic.py
#MuRP MuRs
# HYP_MODELS = ["MuRP"]
MuRP_MODELS = ["MuRP"]
class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity = torch.nn.Embedding(self.sizes[0], self.rank, padding_idx=0)
        self.entity.weight.data = (self.init_size * torch.randn((self.sizes[0], self.rank), dtype=torch.double, device="cuda"))
        self.rel = torch.nn.Embedding(self.sizes[1], self.rank, padding_idx=0)
        self.rel.weight.data = (self.init_size * torch.randn((self.sizes[1], self.rank), dtype=torch.double, device="cuda"))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.sizes[1],
                                        self.rank)), dtype=torch.double, requires_grad=True, device="cuda"))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            rhs=self.entity(queries[:, 2])
            return rhs, self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        score=- hyperbolic_sqdistance(lhs_e, rhs_e, eval_mode)
        return score

class MuRP(BaseH):
    def __init__(self, args):
        super(MuRP, self).__init__(args)
    def get_queries(self, queries):
        u = self.entity.weight[queries[:,0]]
        Ru = self.Wu[queries[:,1]]
        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= 1,
                        u / (torch.norm(u, 2, dim=-1, keepdim=True) - 1e-5), u)
        u_e = p_log_map(u)
        u_W = u_e * Ru
        u_m = p_exp_map(u_W)
        u_m = torch.where(torch.norm(u_m, 2, dim=-1, keepdim=True) >= 1,
                          u_m / (torch.norm(u_m, 2, dim=-1, keepdim=True) - 1e-5), u_m)
        lhs_e=u_m
        return lhs_e,self.bh(queries[:, 0])