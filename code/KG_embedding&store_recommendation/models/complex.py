"""Euclidean Knowledge Graph embedding models where embeddings are in complex space."""
import torch
from torch import nn

from models.base import KGModel

COMPLEX_MODELS = ["ComplEx", "RotatE", "DistMult", "TuckER", "QuatE"]
# COMPLEX_MODELS = ["ComplEx", "RotatE",  "QuatE"]


class BaseC(KGModel):
    """Complex Knowledge Graph Embedding models.

    Attributes:
        embeddings: complex embeddings for entities and relations
    """

    def __init__(self, args):
        """Initialize a Complex KGModel."""
        super(BaseC, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        assert self.rank % 2 == 0, "Complex models require even embedding dimension"
        self.rank = self.rank // 2
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * self.rank, sparse=True)
            for s in self.sizes[:2]
        ])
        self.embeddings[0].weight.data = self.init_size * self.embeddings[0].weight.to(self.data_type)
        self.embeddings[1].weight.data = self.init_size * self.embeddings[1].weight.to(self.data_type)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.embeddings[0].weight, self.bt.weight
        else:
            return self.embeddings[0](queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e = lhs_e[:, :self.rank], lhs_e[:, self.rank:]
        rhs_e = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
        if eval_mode:
            return lhs_e[0] @ rhs_e[0].transpose(0, 1) + lhs_e[1] @ rhs_e[1].transpose(0, 1)
        else:
            return torch.sum(
                lhs_e[0] * rhs_e[0] + lhs_e[1] * rhs_e[1],
                1, keepdim=True
            )

    def get_complex_embeddings(self, queries):
        """Get complex embeddings of queries."""
        head_e = self.embeddings[0](queries[:, 0])
        rel_e = self.embeddings[1](queries[:, 1])
        rhs_e = self.embeddings[0](queries[:, 2])
        head_e = head_e[:, :self.rank], head_e[:, self.rank:]
        rel_e = rel_e[:, :self.rank], rel_e[:, self.rank:]
        rhs_e = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
        return head_e, rel_e, rhs_e

    def get_factors(self, queries):
        """Compute factors for embeddings' regularization."""
        head_e, rel_e, rhs_e = self.get_complex_embeddings(queries)
        head_f = torch.sqrt(head_e[0] ** 2 + head_e[1] ** 2)
        rel_f = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        rhs_f = torch.sqrt(rhs_e[0] ** 2 + rhs_e[1] ** 2)
        return head_f, rel_f, rhs_f


class ComplEx(BaseC):
    """Simple complex model http://proceedings.mlr.press/v48/trouillon16.pdf"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        lhs_e = torch.cat([
            head_e[0] * rel_e[0] - head_e[1] * rel_e[1],
            head_e[0] * rel_e[1] + head_e[1] * rel_e[0]
        ], 1)
        return lhs_e, self.bh(queries[:, 0])


class RotatE(BaseC):
    """Rotations in complex space https://openreview.net/pdf?id=HkgEQnRqYQ"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm
        sin = rel_e[1] / rel_norm
        lhs_e = torch.cat([
            head_e[0] * cos - head_e[1] * sin,
            head_e[0] * sin + head_e[1] * cos
        ], 1)
        return lhs_e, self.bh(queries[:, 0])

class DistMult(BaseC):
    """DistMult model: https://www.microsoft.com/en-us/research/publication/embedding-entities-and-relations-for-learning-and-inference-in-knowledge-bases/"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        # 只取实部（即前半部分）即可视为标准的 real-valued embedding
        head_e = self.embeddings[0](queries[:, 0])[:, :self.rank]
        rel_e = self.embeddings[1](queries[:, 1])[:, :self.rank]
        lhs_e = head_e * rel_e  # element-wise multiplication
        return lhs_e, self.bh(queries[:, 0])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        lhs_e = lhs_e[:, :self.rank]
        rhs_e = rhs_e[:, :self.rank]
        if eval_mode:
            return lhs_e @ rhs_e.transpose(0, 1)
        else:
            return torch.sum(lhs_e * rhs_e, 1, keepdim=True)

    def get_factors(self, queries):
        """Compute factors for embeddings' regularization."""
        head_e = self.embeddings[0](queries[:, 0])[:, :self.rank]
        rel_e = self.embeddings[1](queries[:, 1])[:, :self.rank]
        rhs_e = self.embeddings[0](queries[:, 2])[:, :self.rank]
        return head_e.abs(), rel_e.abs(), rhs_e.abs()

class TuckER(KGModel):
    """TuckER: Tucker decomposition for knowledge graph embedding
    Paper: https://arxiv.org/abs/1806.07297
    """

    def __init__(self, args):
        super(TuckER, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias, args.init_size)
        self.entity_dim = args.rank
        self.relation_dim = args.rank  # 通常设置相同
        self.emb_e = nn.Embedding(args.sizes[0], self.entity_dim)
        self.emb_r = nn.Embedding(args.sizes[1], self.relation_dim)

        # 核心张量 W: 实现为线性映射
        self.W = nn.Parameter(torch.randn(self.relation_dim, self.entity_dim, self.entity_dim) * args.init_size)

        # 初始化权重
        nn.init.xavier_uniform_(self.emb_e.weight)
        nn.init.xavier_uniform_(self.emb_r.weight)

    def get_queries(self, queries):
        head = self.emb_e(queries[:, 0])  # [batch_size, d_e]
        rel = self.emb_r(queries[:, 1])   # [batch_size, d_r]
        lhs_e = self._tucker_projection(head, rel)
        return lhs_e, self.bh(queries[:, 0])

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.emb_e.weight, self.bt.weight
        else:
            return self.emb_e(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        if eval_mode:
            return lhs_e @ rhs_e.transpose(0, 1)
        else:
            return torch.sum(lhs_e * rhs_e, dim=1, keepdim=True)

    def _tucker_projection(self, head, rel):
        # head: [B, d_e], rel: [B, d_r]
        W = torch.einsum('br, rmn -> bmn', rel, self.W)  # [B, d_e, d_e]
        lhs_e = torch.einsum('bi, bij -> bj', head, W)   # [B, d_e]
        return lhs_e

    def get_factors(self, queries):
        head = self.emb_e(queries[:, 0])
        rel = self.emb_r(queries[:, 1])
        tail = self.emb_e(queries[:, 2])
        return head.abs(), rel.abs(), tail.abs()


class QuatE(KGModel):
    def __init__(self, args):
        super().__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias, args.init_size)
        self.rank = args.rank  # d
        self.emb_e = nn.Embedding(args.sizes[0], 4 * self.rank)
        self.emb_r = nn.Embedding(args.sizes[1], 4 * self.rank)

        nn.init.xavier_uniform_(self.emb_e.weight)
        nn.init.xavier_uniform_(self.emb_r.weight)

    def get_queries(self, queries):
        h = self.emb_e(queries[:, 0])
        r = self.emb_r(queries[:, 1])
        lhs = self.quaternion_mult(h, r)
        return lhs, self.bh(queries[:, 0])

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.emb_e.weight, self.bt.weight
        else:
            return self.emb_e(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        if eval_mode:
            return lhs_e @ rhs_e.transpose(0, 1)
        else:
            return torch.sum(lhs_e * rhs_e, dim=1, keepdim=True)

    def quaternion_mult(self, h, r):
        """四元数乘法：h ⊗ r"""
        h_r, h_i, h_j, h_k = torch.chunk(h, 4, dim=1)
        r_r, r_i, r_j, r_k = torch.chunk(r, 4, dim=1)

        A = h_r * r_r - h_i * r_i - h_j * r_j - h_k * r_k
        B = h_r * r_i + h_i * r_r + h_j * r_k - h_k * r_j
        C = h_r * r_j - h_i * r_k + h_j * r_r + h_k * r_i
        D = h_r * r_k + h_i * r_j - h_j * r_i + h_k * r_r

        return torch.cat([A, B, C, D], dim=1)

    def get_factors(self, queries):
        h = self.emb_e(queries[:, 0])
        r = self.emb_r(queries[:, 1])
        t = self.emb_e(queries[:, 2])
        return h.abs(), r.abs(), t.abs()


class QuatE(KGModel):
    def __init__(self, args):
        super().__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias, args.init_size)
        self.rank = args.rank  # d
        self.emb_e = nn.Embedding(args.sizes[0], 4 * self.rank)
        self.emb_r = nn.Embedding(args.sizes[1], 4 * self.rank)

        nn.init.xavier_uniform_(self.emb_e.weight)
        nn.init.xavier_uniform_(self.emb_r.weight)

    def get_queries(self, queries):
        h = self.emb_e(queries[:, 0])
        r = self.emb_r(queries[:, 1])
        lhs = self.quaternion_mult(h, r)
        return lhs, self.bh(queries[:, 0])

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.emb_e.weight, self.bt.weight
        else:
            return self.emb_e(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        if eval_mode:
            return lhs_e @ rhs_e.transpose(0, 1)
        else:
            return torch.sum(lhs_e * rhs_e, dim=1, keepdim=True)

    def quaternion_mult(self, h, r):
        h_r, h_i, h_j, h_k = torch.chunk(h, 4, dim=1)
        r_r, r_i, r_j, r_k = torch.chunk(r, 4, dim=1)

        A = h_r * r_r - h_i * r_i - h_j * r_j - h_k * r_k
        B = h_r * r_i + h_i * r_r + h_j * r_k - h_k * r_j
        C = h_r * r_j - h_i * r_k + h_j * r_r + h_k * r_i
        D = h_r * r_k + h_i * r_j - h_j * r_i + h_k * r_r

        return torch.cat([A, B, C, D], dim=1)

    def get_factors(self, queries):
        h = self.emb_e(queries[:, 0])
        r = self.emb_r(queries[:, 1])
        t = self.emb_e(queries[:, 2])
        return h.abs(), r.abs(), t.abs()
