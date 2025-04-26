import os
import pickle
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torchdrug.layers import functional



def safe_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


class EmerGNN(nn.Module):
    """
    完全复现原始 EmerGNN 的单专家实现
    支持两种实体初始特征:
        • E : 可学习随机嵌入
        • M : 固定 Morgan 指纹  (data/DB_molecular_feats.pkl)
    """
    def __init__(self, eval_ent, eval_rel, args):
        super(EmerGNN, self).__init__()
        self.eval_ent = eval_ent
        self.eval_rel = eval_rel
        self.args = args
        self.all_ent = args.all_ent
        self.all_rel = args.all_rel
        self.L = args.length
        all_rel = 2 * args.all_rel + 1

        # ---------- 药物初始表征 ----------
        if args.feat.upper() == "M":
            # 兼容旧目录(data-DB)与新目录(data)
            cand = [
                os.path.join(args.task_dir, "data-DB/DB_molecular_feats.pkl"),
                os.path.join(args.task_dir, "data/DB_molecular_feats.pkl"),
                "data-DB/DB_molecular_feats.pkl",
                "data/DB_molecular_feats.pkl",
            ]
            for p in cand:
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        mfeat_list = pickle.load(f, encoding="utf-8")["Morgan_Features"]
                    break
            else:
                raise FileNotFoundError(
                    "Cannot locate DB_molecular_feats.pkl; searched:\n" + "\n".join(cand)
                )

            # ---- 修复：list → ndarray(float32) → tensor ----
            # mfeat_list 是长度为 N 的列表，每个元素是长度 1024 的 numpy.ndarray
            mfeat_arr = np.stack(mfeat_list, axis=0).astype(np.float32)  # [N,1024]
            mf = torch.from_numpy(mfeat_arr)                             # Tensor
            # ------------------------------------------------

            self.ent_kg = nn.Parameter(mf, requires_grad=False)
            self.Went = nn.Linear(1024, args.n_dim, bias=False)
            self.Wr = nn.Linear(2 * args.n_dim, eval_rel)
        else:  # 'E'
            self.ent_kg = nn.Embedding(eval_ent, args.n_dim)
            self.Wr = nn.Linear(4 * args.n_dim, eval_rel)

        # ---------- 网络其余部分 ----------
        self.rel_kg = nn.ModuleList([nn.Embedding(all_rel, args.n_dim) for _ in range(self.L)])
        self.n_dim = args.n_dim
        self.linear = nn.ModuleList([nn.Linear(args.n_dim, args.n_dim) for _ in range(self.L)])
        self.W = nn.Linear(args.n_dim, 1)
        self.act = nn.ReLU()

        self.relation_linear = nn.ModuleList([nn.Linear(2 * args.n_dim, 5) for _ in range(self.L)])
        self.attn_relation = nn.ModuleList([nn.Linear(5, all_rel) for _ in range(self.L)])

        self._init_weight()

    def _init_weight(self):
        for p in self.parameters():
            if p.requires_grad and p.data.ndim > 1:
                nn.init.xavier_uniform_(p.data)

    # ---------------------------  forward  ---------------------------
    def enc_ht(self, head, tail, KG, visualize: bool = False):
        if self.args.feat.upper() == "M":
            head_embed = self.Went(self.ent_kg[head])
            tail_embed = self.Went(self.ent_kg[tail])
        else:
            head_embed = self.ent_kg(head)
            tail_embed = self.ent_kg(tail)

        n_ent = self.all_ent

        # u → v
        hiddens = safe_cuda(torch.zeros(n_ent, len(head), self.n_dim))
        hiddens[head, safe_cuda(torch.arange(len(head)))] = head_embed
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)

        for l in range(self.L):
            hiddens = hiddens.view(n_ent, -1)
            rel_w = torch.sigmoid(
                self.attn_relation[l](self.act(self.relation_linear[l](ht_embed)))
            ).unsqueeze(2)
            rel_embed = self.rel_kg[l].weight
            rel_in = (rel_w * rel_embed).view(head_embed.size(0), -1, self.n_dim)
            rel_in = rel_in.transpose(0, 1).flatten(1)
            hiddens = functional.generalized_rspmm(KG, rel_in, hiddens, sum="add", mul="mul")
            hiddens = self.act(self.linear[l](hiddens.view(n_ent * len(head), -1)))
        tail_hid = hiddens.view(n_ent, len(tail), -1)[tail, safe_cuda(torch.arange(len(tail)))]

        # v → u
        # 重新创建 3-D 张量，形状 (n_ent, |tail|, d)
        hiddens = safe_cuda(torch.zeros(n_ent, len(head), self.n_dim))
        hiddens[tail, safe_cuda(torch.arange(len(tail)))] = tail_embed
        for l in range(self.L):
            hiddens = hiddens.view(n_ent, -1)
            rel_w = torch.sigmoid(
                self.attn_relation[l](self.act(self.relation_linear[l](ht_embed)))
            ).unsqueeze(2)
            rel_embed = self.rel_kg[l].weight
            rel_in = (rel_w * rel_embed).view(head_embed.size(0), -1, self.n_dim)
            rel_in = rel_in.transpose(0, 1).flatten(1)
            hiddens = functional.generalized_rspmm(KG, rel_in, hiddens, sum="add", mul="mul")
            hiddens = self.act(self.linear[l](hiddens.view(n_ent * len(head), -1)))
        head_hid = hiddens.view(n_ent, len(head), -1)[head, safe_cuda(torch.arange(len(head)))]

        if self.args.feat.upper() == "M":
            embeddings = torch.cat([head_hid, tail_hid], dim=1)
        else:
            embeddings = torch.cat([head_embed, tail_embed, head_hid, tail_hid], dim=1)
        return embeddings

    def enc_r(self, ht_embed):
        return self.Wr(ht_embed)

    def get_attention_weights(self, head, tail, KG):
        if self.args.feat.upper() == "M":
            head_embed = self.Went(self.ent_kg[head])
            tail_embed = self.Went(self.ent_kg[tail])
        else:
            head_embed = self.ent_kg(head)
            tail_embed = self.ent_kg(tail)
        ht_embed = torch.cat([head_embed, tail_embed], -1)
        outs = []
        for l in range(self.L):
            rel_w = torch.sigmoid(
                self.attn_relation[l](self.act(self.relation_linear[l](ht_embed)))
            )
            outs.append(rel_w.cpu().numpy())
        return outs


# -------------------------  MoE 部分  -------------------------
class _SingleExpert(EmerGNN):
    """在多专家场景下共享实体嵌入"""
    def __init__(self, ent_emb: nn.Embedding, eval_ent, eval_rel, args):
        super().__init__(eval_ent, eval_rel, args)
        self.ent_kg = ent_emb  # 重用共享实体嵌入


class MoE_EmerGNN(nn.Module):
    """
    Multi-Expert EmerGNN (soft-gate / top-k gate)
    """
    def __init__(self, kg_names: List[str], eval_ent: int, eval_rel: int, args):
        super().__init__()
        self.kg_names = kg_names
        self.n_exp = len(kg_names)
        self.eval_rel = eval_rel
        self.args = args

        # ----- 共享实体嵌入 -----
        if args.feat.upper() == "M":
            with open("data/DB_molecular_feats.pkl", "rb") as f:
                feats = pickle.load(f, encoding="utf-8")["Morgan_Features"]
            ent_emb_fixed = torch.FloatTensor(feats)
            self.ent_emb = nn.Embedding.from_pretrained(ent_emb_fixed, freeze=True)
            self._went = nn.Linear(1024, args.n_dim)
        else:
            self.ent_emb = nn.Embedding(eval_ent, args.n_dim)

        # ----- 每个专家 -----
        self.experts = nn.ModuleDict(
            {name: _SingleExpert(self.ent_emb, eval_ent, eval_rel, args) for name in kg_names}
        )

        # ----- 门控 -----
        gate_in = 2 * args.n_dim
        h = args.gate_hidden
        if h > 0:
            self.gate = nn.Sequential(nn.Linear(gate_in, h), nn.ReLU(), nn.Linear(h, self.n_exp))
        else:
            self.gate = nn.Linear(gate_in, self.n_exp)
        self.gate_type = args.gate_type
        self.top_k = args.gate_topk

    def _get_plain_ht(self, h, t):
        if self.args.feat.upper() == "M":
            h_emb = self._went(self.ent_emb(h))
            t_emb = self._went(self.ent_emb(t))
        else:
            h_emb = self.ent_emb(h)
            t_emb = self.ent_emb(t)
        return torch.cat([h_emb, t_emb], -1)  # [B, 2d]

    def forward(self, heads, tails, ddi_adj, kg_dict: Dict[str, torch.Tensor]):
        # ---- gate α ----
        logits = self.gate(self._get_plain_ht(heads, tails))  # [B,K]
        if self.gate_type == "soft":
            α = torch.softmax(logits, -1)
        else:
            topk_idx = torch.topk(logits, self.top_k, -1).indices
            α = torch.zeros_like(logits)
            α.scatter_(1, topk_idx, 1.0 / self.top_k)

        # ---- per-expert logits ----
        logits_list = []
        for name in self.kg_names:
            kg_i = ddi_adj + kg_dict[name]
            ht_emb = self.experts[name].enc_ht(heads, tails, kg_i)
            logits_list.append(self.experts[name].enc_r(ht_emb))
        logits_stack = torch.stack(logits_list, 1)  # [B,K,R]

        return (α.unsqueeze(-1) * logits_stack).sum(1)  # [B,R]


# ---- 旧版 MoE（若其他代码依赖） ----
class MoE(nn.Module):
    def __init__(self, n_ent, n_rel, n_dim, n_layer, n_head, n_expert, n_batch, dropout, lamb, device):
        super().__init__()
        self.experts = nn.ModuleList(
            [EmerGNN(n_ent, n_rel, n_dim, n_layer, n_head, n_batch, dropout, lamb, device)]
            * n_expert
        )
        self.gate = nn.Sequential(
            nn.Linear(n_dim * 2, n_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_dim, n_expert),
            nn.Softmax(dim=-1),
        )

    def forward(self, h, t, ddi_adj, kg_dict):
        h_emb = self.experts[0].ent_emb(h)
        t_emb = self.experts[0].ent_emb(t)
        α = self.gate(torch.cat([h_emb, t_emb], -1))
        exp_logits = []
        for i, exp in enumerate(self.experts):
            kg = ddi_adj + kg_dict[list(kg_dict.keys())[i]]
            exp_logits.append(exp(h, t, kg))
        exp_logits = torch.stack(exp_logits, 1)
        return (α.unsqueeze(-1) * exp_logits).sum(1)

    def get_ent_emb(self):
        return self.experts[0].ent_emb.weight.data
