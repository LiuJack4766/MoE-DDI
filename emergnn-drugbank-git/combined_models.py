import os
import pickle
import json
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torchdrug.layers import functional


def load_unified_relation_space():
    """加载统一的关系空间，返回最大关系ID"""
    paths = [
        "data/relation2id.json",
        "relation2id.json",
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                relation2id = json.load(f)
            max_id = max(int(k) for k in relation2id.keys())
            print(f"成功从{path}加载统一关系空间，最大关系ID: {max_id}")
            return max_id
    # 如果找不到文件，使用默认值
    print("警告: 无法找到relation2id.json文件，使用默认最大关系ID: 109")
    return 109  # 默认最大关系ID


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
        
        # 强制使用统一的关系空间
        max_relation_id = load_unified_relation_space()
        self.all_rel = max_relation_id + 1  # 强制使用统一的关系空间
        
        self.L = args.length
        all_rel = 2 * self.all_rel + 1  # 使用统一的all_rel值

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
            # 始终对齐 KG 的第三维度 (2*R+1)
            target_dim = KG.shape[2]
            # 1. 对齐rel_embed
            if rel_embed.shape[0] != target_dim:
                if rel_embed.shape[0] < target_dim:  # pad
                    pad = target_dim - rel_embed.shape[0]
                    rel_embed = F.pad(rel_embed, (0, 0, 0, pad))
                else:  # truncate
                    rel_embed = rel_embed[:target_dim]
            # 2. 对齐rel_w
            if rel_w.shape[1] != target_dim:
                if rel_w.shape[1] < target_dim:  # pad
                    pad = target_dim - rel_w.shape[1]
                    rel_w = F.pad(rel_w, (0, 0, 0, pad))
                else:  # truncate
                    rel_w = rel_w[:, :target_dim]
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
            # 始终对齐 KG 的第三维度 (2*R+1)
            target_dim = KG.shape[2]
            # 1. 对齐rel_embed
            if rel_embed.shape[0] != target_dim:
                if rel_embed.shape[0] < target_dim:  # pad
                    pad = target_dim - rel_embed.shape[0]
                    rel_embed = F.pad(rel_embed, (0, 0, 0, pad))
                else:  # truncate
                    rel_embed = rel_embed[:target_dim]
            # 2. 对齐rel_w
            if rel_w.shape[1] != target_dim:
                if rel_w.shape[1] < target_dim:  # pad
                    pad = target_dim - rel_w.shape[1]
                    rel_w = F.pad(rel_w, (0, 0, 0, pad))
                else:  # truncate
                    rel_w = rel_w[:, :target_dim]
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
            
            # 始终对齐 KG 的第三维度 (2*R+1)
            target_dim = KG.shape[2]
            if rel_w.shape[1] != target_dim:  # 修正：检查第二维（关系数量）
                if rel_w.shape[1] < target_dim:  # pad
                    pad = target_dim - rel_w.shape[1]
                    # 需要根据实际维度调整padding方式
                    rel_w = F.pad(rel_w, (0, pad))  # 对第二维进行padding
                else:  # truncate
                    rel_w = rel_w[:, :target_dim]  # 对第二维进行截断
            
            outs.append(rel_w.cpu().detach().numpy())
        return outs


# -------------------------  MoE 部分  -------------------------
class _SingleExpert(EmerGNN):
    """在多专家场景下共享实体嵌入"""
    def __init__(self, ent_emb: nn.Embedding, eval_ent, eval_rel, args):
        # 确保使用统一的关系空间
        max_relation_id = load_unified_relation_space()
        args.all_rel = max_relation_id + 1  # 强制覆盖args中的all_rel
        
        super().__init__(eval_ent, eval_rel, args)
        # 不直接赋值ent_kg，而是保存共享嵌入的引用
        self._shared_emb = ent_emb
        # 确保所有组件在正确的设备上
        if args.feat.upper() == "M":
            self.Went = nn.Linear(1024, args.n_dim, bias=False)
            # 将Went移动到与ent_emb相同的设备
            if next(ent_emb.parameters()).is_cuda:
                self.Went = self.Went.cuda()
                # 确保所有其他组件也在CUDA上
                for module in [self.linear, self.relation_linear, self.attn_relation, self.rel_kg]:
                    for layer in module:
                        layer.cuda()
        
    def enc_ht(self, head, tail, KG, visualize: bool = False):
        """重写enc_ht方法，使用共享嵌入代替self.ent_kg"""
        if self.args.feat.upper() == "M":
            # 确保输入在正确的设备上
            if next(self._shared_emb.parameters()).is_cuda:
                head = head.cuda()
                tail = tail.cuda()
                KG = KG.cuda()
            head_embed = self.Went(self._shared_emb(head))
            tail_embed = self.Went(self._shared_emb(tail))
        else:
            head_embed = self._shared_emb(head)
            tail_embed = self._shared_emb(tail)

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
            # 始终对齐 KG 的第三维度 (2*R+1)
            target_dim = KG.shape[2]
            # 1. 对齐rel_embed
            if rel_embed.shape[0] != target_dim:
                if rel_embed.shape[0] < target_dim:  # pad
                    pad = target_dim - rel_embed.shape[0]
                    rel_embed = F.pad(rel_embed, (0, 0, 0, pad))
                else:  # truncate
                    rel_embed = rel_embed[:target_dim]
            # 2. 对齐rel_w
            if rel_w.shape[1] != target_dim:
                if rel_w.shape[1] < target_dim:  # pad
                    pad = target_dim - rel_w.shape[1]
                    rel_w = F.pad(rel_w, (0, 0, 0, pad))
                else:  # truncate
                    rel_w = rel_w[:, :target_dim]
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
            # 始终对齐 KG 的第三维度 (2*R+1)
            target_dim = KG.shape[2]
            # 1. 对齐rel_embed
            if rel_embed.shape[0] != target_dim:
                if rel_embed.shape[0] < target_dim:  # pad
                    pad = target_dim - rel_embed.shape[0]
                    rel_embed = F.pad(rel_embed, (0, 0, 0, pad))
                else:  # truncate
                    rel_embed = rel_embed[:target_dim]
            # 2. 对齐rel_w
            if rel_w.shape[1] != target_dim:
                if rel_w.shape[1] < target_dim:  # pad
                    pad = target_dim - rel_w.shape[1]
                    rel_w = F.pad(rel_w, (0, 0, 0, pad))
                else:  # truncate
                    rel_w = rel_w[:, :target_dim]
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
        
    def get_attention_weights(self, head, tail, KG):
        """重写get_attention_weights方法，使用共享嵌入"""
        if self.args.feat.upper() == "M":
            head_embed = self.Went(self._shared_emb(head))
            tail_embed = self.Went(self._shared_emb(tail))
        else:
            head_embed = self._shared_emb(head)
            tail_embed = self._shared_emb(tail)
        ht_embed = torch.cat([head_embed, tail_embed], -1)
        outs = []
        for l in range(self.L):
            rel_w = torch.sigmoid(
                self.attn_relation[l](self.act(self.relation_linear[l](ht_embed)))
            )
            
            # 始终对齐 KG 的第三维度 (2*R+1)
            target_dim = KG.shape[2]
            if rel_w.shape[1] != target_dim:  # 修正：检查第二维（关系数量）
                if rel_w.shape[1] < target_dim:  # pad
                    pad = target_dim - rel_w.shape[1]
                    # 需要根据实际维度调整padding方式
                    rel_w = F.pad(rel_w, (0, pad))  # 对第二维进行padding
                else:  # truncate
                    rel_w = rel_w[:, :target_dim]  # 对第二维进行截断
            
            outs.append(rel_w.cpu().detach().numpy())
        return outs


class MoE_EmerGNN(nn.Module):
    """
    Multi-Expert EmerGNN (soft-gate / top-k gate)
    """
    def __init__(self, kg_names: List[str], eval_ent: int, eval_rel: int, args):
        super().__init__()
        self.kg_names = kg_names
        self.n_exp = len(kg_names)
        self.eval_rel = eval_rel
        self.eval_ent = eval_ent
        self.args = args
        
        # 确保使用统一的关系空间
        max_relation_id = load_unified_relation_space()
        args.all_rel = max_relation_id + 1  # 强制覆盖args中的all_rel

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
        self.experts = nn.ModuleDict()
        for name in kg_names:
            # 先创建一个临时专家来获取维度信息
            temp_expert = _SingleExpert(self.ent_emb, eval_ent, eval_rel, args)
            self.experts[name] = temp_expert

        # ----- 门控 -----
        gate_in = 2 * args.n_dim
        h = args.gate_hidden
        if h > 0:
            self.gate = nn.Sequential(nn.Linear(gate_in, h), nn.ReLU(), nn.Linear(h, self.n_exp))
        else:
            self.gate = nn.Linear(gate_in, self.n_exp)
        self.gate_type = args.gate_type
        self.top_k = args.gate_topk
        
    def load_expert(self, expert_name, checkpoint_path):
        """从检查点加载单个专家模型的参数"""
        if expert_name not in self.kg_names:
            raise ValueError(f"未知专家名称: {expert_name}，可用专家: {self.kg_names}")
            
        if not os.path.exists(checkpoint_path):
            print(f"警告: 找不到检查点 {checkpoint_path}")
            return False
            
        # 加载单专家检查点
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # ---------- 1. 取目标专家的当前参数 ----------
        target_expert = self.experts[expert_name]
        target_sd = target_expert.state_dict()
        
        # ---------- 2. 生成"尺寸对齐"的新 state_dict ----------
        aligned_sd = {}
        prefix_len = len("model.")  # 旧 checkpoint 是否带 "model." 前缀
        
        for k_ckpt, v_ckpt in state_dict.items():
            # 去掉可能存在的 "model." 前缀，并加上 experts.{name}. 这一步后只剩裸键
            k_clean = k_ckpt[prefix_len:] if k_ckpt.startswith("model.") else k_ckpt
            
            # 跳过 ent_kg（已共享）
            if k_clean.startswith("ent_kg"):
                continue
                
            # 只留下属于当前专家且模型里有的键
            if k_clean not in target_sd:
                continue
                
            v_tgt = target_sd[k_clean]
            if v_ckpt.shape != v_tgt.shape:
                # ---- 特殊处理注意力模块的权重 ----
                if "attn_relation" in k_clean and "weight" in k_clean:
                    print(f"调整注意力权重维度: {k_clean}: {v_ckpt.shape} -> {v_tgt.shape}")
                    # 对于权重矩阵，形状是[out_features, in_features]
                    if v_ckpt.shape[0] != v_tgt.shape[0]:  # 输出维度不匹配
                        # 创建新权重矩阵
                        new_weight = torch.zeros(v_tgt.shape, dtype=v_ckpt.dtype)
                        # 复制共同部分
                        min_out = min(v_ckpt.shape[0], v_tgt.shape[0])
                        new_weight[:min_out, :] = v_ckpt[:min_out, :]
                        v_ckpt = new_weight
                
                # ---- 通用处理：按第一维 pad / truncate ----
                elif v_ckpt.shape[0] != v_tgt.shape[0]:  # 第一维不匹配
                    dim0_min = min(v_ckpt.shape[0], v_tgt.shape[0])
                    v_adj = v_ckpt[:dim0_min].clone()
                    if v_adj.shape[0] < v_tgt.shape[0]:  # 需要 pad
                        pad_rows = v_tgt.shape[0] - v_adj.shape[0]
                        pad_shape = (pad_rows, *v_adj.shape[1:])
                        v_adj = torch.cat([v_adj,
                                         torch.zeros(pad_shape, dtype=v_adj.dtype)], dim=0)
                    v_ckpt = v_adj
                
            aligned_sd[k_clean] = v_ckpt
            
        # ---------- 3. 把处理后的字典 load 进子专家 ----------
        missing, unexpected = target_expert.load_state_dict(aligned_sd, strict=False)
        
        print(f">> 已加载专家 {expert_name} 参数")
        if len(missing) > 0:
            relevant_missing = [k for k in missing if not k.startswith("ent_kg")]
            if relevant_missing:
                print(f">> 注意: 专家 {expert_name} 有未匹配参数: {len(relevant_missing)}个")
                
        return True

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
        logits = self.gate(self._get_plain_ht(heads, tails))          # [B, K]
        if self.gate_type == "soft":
            α = torch.softmax(logits, -1)
        else:
            idx = torch.topk(logits, self.top_k, -1).indices
            α = torch.zeros_like(logits)
            α.scatter_(1, idx, 1.0 / self.top_k)

        # ---- per-expert logits ----
        add_ddi = self.training       # True ↔ model.train(); False ↔ model.eval()
        logits_list = []
        for name in self.kg_names:
            kg_i = kg_dict[name] if not add_ddi else (ddi_adj + kg_dict[name])
            ht_emb = self.experts[name].enc_ht(heads, tails, kg_i)
            logits_list.append(self.experts[name].enc_r(ht_emb))

        logits_stack = torch.stack(logits_list, 1)                     # [B, K, R]
        return (α.unsqueeze(-1) * logits_stack).sum(1)                 # [B, R]
        
    def debug_attention_weights(self):
        """打印所有专家的注意力权重矩阵维度，用于调试"""
        print("\n========== 专家注意力权重矩阵维度检查 ==========")
        # 创建一个简单的测试输入
        test_head = torch.LongTensor([0])
        test_tail = torch.LongTensor([1])
        dummy_kg = torch.zeros((self.eval_ent, self.eval_ent, 2 * self.args.all_rel + 1))
        
        # 检查每个专家的注意力权重维度
        for name, expert in self.experts.items():
            for l in range(expert.L):
                # 获取原始权重矩阵维度
                weight_shape = expert.attn_relation[l].weight.shape
                # 获取输出维度
                if self.args.feat.upper() == "M":
                    head_embed = expert.Went(self.ent_emb(test_head))
                    tail_embed = expert.Went(self.ent_emb(test_tail))
                else:
                    head_embed = self.ent_emb(test_head)
                    tail_embed = self.ent_emb(test_tail)
                ht_embed = torch.cat([head_embed, tail_embed], -1)
                rel_w = expert.attn_relation[l](expert.act(expert.relation_linear[l](ht_embed)))
                
                print(f"专家 {name} - 层 {l}:")
                print(f"  权重矩阵: {weight_shape}")
                print(f"  输出维度: {rel_w.shape}")
                print(f"  KG关系维度: {dummy_kg.shape[2]}")
                
        # 检查专家间rel_kg权重维度
        print("\n---------- 专家关系嵌入维度 ----------")
        for name, expert in self.experts.items():
            for l in range(expert.L):
                rel_embed_shape = expert.rel_kg[l].weight.shape
                print(f"专家 {name} - 层 {l}: {rel_embed_shape}")
        
        print("========================================")


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
