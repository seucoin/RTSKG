"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn

from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection
import os

from torch.nn.init import xavier_normal_


EUC_MODELS = ["TransE", "DisMult","TuckER","CP", "MurE", "RotE", "RefE", "AttE"]



class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score


class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.sim = "dist"

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class CP(BaseE):
    """Canonical tensor decomposition https://arxiv.org/pdf/1806.07297.pdf"""

    def __init__(self, args):
        super(CP, self).__init__(args)
        self.sim = "dot"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        return self.entity(queries[:, 0]) * self.rel(queries[:, 1]), self.bh(queries[:, 0])


class MurE(BaseE):
    """Diagonal scaling https://arxiv.org/pdf/1905.09791.pdf"""

    def __init__(self, args):
        super(MurE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = self.rel_diag(queries[:, 1]) * self.entity(queries[:, 0]) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

class MurS(BaseE):
    """Diagonal scaling without translation (MurS)"""

    def __init__(self, args):
        super(MurS, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        lhs_e = self.rel_diag(queries[:, 1]) * self.entity(queries[:, 0])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases



class RotE(BaseE):
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(RotE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = givens_rotations(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

class MurP(BaseE):
    """Full relation projection (MurP)"""

    def __init__(self, args):
        super(MurP, self).__init__(args)
        self.rel_proj = nn.Embedding(self.sizes[1], self.rank * self.rank)
        self.rel_proj.weight.data = (
            self.init_size * torch.randn((self.sizes[1], self.rank * self.rank), dtype=self.data_type)
        )
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        head_e = self.entity(queries[:, 0])  # [B, d]
        proj_r = self.rel_proj(queries[:, 1])  # [B, d*d]
        proj_r = proj_r.view(-1, self.rank, self.rank)  # [B, d, d]

        lhs_e = torch.bmm(proj_r, head_e.unsqueeze(-1)).squeeze(-1)  # [B, d]
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class RefE(BaseE):
    """Euclidean 2x2 Givens reflections"""

    def __init__(self, args):
        super(RefE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        rel = self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs + rel, lhs_biases


class AttE(BaseE):
    """Euclidean attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttE, self).__init__(args)
        self.sim = "dist"

        # reflection
        self.ref = nn.Embedding(self.sizes[1], self.rank)
        self.ref.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # rotation
        self.rot = nn.Embedding(self.sizes[1], self.rank)
        self.rot.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_reflection_queries(self, queries):
        lhs_ref_e = givens_reflection(
            self.ref(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_ref_e

    def get_rotation_queries(self, queries):
        lhs_rot_e = givens_rotations(
            self.rot(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_rot_e

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs_ref_e = self.get_reflection_queries(queries).view((-1, 1, self.rank))
        lhs_rot_e = self.get_rotation_queries(queries).view((-1, 1, self.rank))

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])
        return lhs_e, self.bh(queries[:, 0])

class CP(BaseE):
    """Canonical tensor decomposition https://arxiv.org/pdf/1806.07297.pdf"""

    def __init__(self, args):
        super(CP, self).__init__(args)
        self.sim = "dot"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        return self.entity(queries[:, 0]) * self.rel(queries[:, 1]), self.bh(queries[:, 0])


class DisMult(BaseE):
    "https://arxiv.org/pdf/1412.6575"
    def __init__(self,args):
        super(DisMult, self).__init__(args)
        self.sim = "dot"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        return self.entity(queries[:, 0]) * self.rel(queries[:, 1]), self.bh(queries[:, 0])

class TuckER(BaseE):
    '''https://github.com/ibalazevic/TuckER/blob/master/model.py'''
    def __init__(self, args):
        super(TuckER, self).__init__(args)
        self.sim = "dot"
        # 核心张量 W，假设 d1（实体维度）= rank, d2（关系维度）可配置
        self.d1 = args.rank  # 实体嵌入维度
        self.d2 = getattr(args, 'd2', args.rank)  # 关系嵌入维度，默认为 rank
        self.W = nn.Parameter(
            torch.tensor(
                np.random.uniform(-1, 1, (self.d2, self.d1, self.d1)),
                dtype=torch.float32,
                device="cuda" if torch.cuda.is_available() else "cpu",
                requires_grad=True
            )
        )
        # 批归一化和 dropout
        self.bn0 = nn.BatchNorm1d(self.d1)
        self.bn1 = nn.BatchNorm1d(self.d1)
        self.input_dropout = nn.Dropout(getattr(args, 'input_dropout', 0.2))
        self.hidden_dropout1 = nn.Dropout(getattr(args, 'hidden_dropout1', 0.3))
        self.hidden_dropout2 = nn.Dropout(getattr(args, 'hidden_dropout2', 0.3))
        # 初始化嵌入
        xavier_normal_(self.entity.weight.data)
        xavier_normal_(self.rel.weight.data)


    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        # 获取头实体嵌入
        head_e = self.entity(queries[:, 0])  # 形状: (N, d1)
        # 获取关系嵌入
        rel_e = self.rel(queries[:, 1])      # 形状: (N, d2)
        # 获取头实体偏置
        lhs_biases = self.bh(queries[:, 0])  # 形状: (N, 1)

        head_e=head_e.float()
        rel_e=rel_e.float()

        head_e = self.bn0(head_e)            # 批归一化
        head_e = self.input_dropout(head_e)  # Dropout
        head_e = head_e.view(-1, 1, head_e.size(1))  # 形状: (N, 1, d1)

        # 计算关系特定的变换矩阵 W_r = W ×_2 r
        W_mat = torch.mm(rel_e, self.W.view(rel_e.size(1), -1))  # 形状: (N, d1*d1)
        W_mat = W_mat.view(-1, head_e.size(2), head_e.size(2))   # 形状: (N, d1, d1)
        W_mat = self.hidden_dropout1(W_mat)   # Dropout
        # 计算查询嵌入 x' = e_h · W_r
        x = torch.bmm(head_e, W_mat)         # 形状: (N, 1, d1)
        x = x.view(-1, head_e.size(2))       # 形状: (N, d1)
        x = self.bn1(x)                      # 批归一化
        lhs_e = self.hidden_dropout2(x)          # Dropout
        lhs_e=lhs_e.double()
        return lhs_e, lhs_biases