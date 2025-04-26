import os
import torch
import numpy as np
import random
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def safe_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


# ======================================================================
#                               基础 DataLoader
# ======================================================================
class DataLoader:
    def __init__(self, params, saved_relation2id: Dict[int, int] = None):
        self.task_dir = params.task_dir
        self.dataset = params.dataset

        ddi_paths = {
            "train": os.path.join(self.task_dir, f"data/{self.dataset}/train_ddi.txt"),
            "valid": os.path.join(self.task_dir, f"data/{self.dataset}/valid_ddi.txt"),
            "test":  os.path.join(self.task_dir, f"data/{self.dataset}/test_ddi.txt"),
        }

        # 处理 DDI 三元组
        self.process_files_ddi(ddi_paths, saved_relation2id)
        # 加载实体 / 关系名称映射（仅用于可视化）
        self.load_ent_id()

    # ------------------------------------------------------------------
    def process_files_ddi(self, paths: Dict[str, str], saved_relation2id=None):
        entity2id = {}
        relation2id = {} if saved_relation2id is None else saved_relation2id

        self.triplets = {}
        self.train_ent = set()

        for split, file_path in paths.items():
            data = []
            with open(file_path) as f:
                lines = [ln.split() for ln in f.read().split("\n") if ln.strip()]

            for h, t, r in lines:
                h, t, r = int(h), int(t), int(r)
                entity2id.setdefault(h, h)
                entity2id.setdefault(t, t)
                if saved_relation2id is None:
                    relation2id.setdefault(r, r)

                if split == "train":
                    self.train_ent.update([h, t])

                data.append([h, t, r])

            self.triplets[split] = np.array(data, dtype="int")

        self.entity2id = entity2id
        self.relation2id = relation2id
        self.eval_ent = max(self.entity2id.keys()) + 1
        self.eval_rel = 86  # DDI 关系数量

        # -------- 打印 DDI 统计 --------
        print(
            f"DDI triplets        : "
            f"Train-{len(self.triplets['train'])} "
            f"Valid-{len(self.triplets['valid'])} "
            f"Test-{len(self.triplets['test'])}"
        )

    # ------------------------------------------------------------------
    def load_ent_id(self):
        id2entity, id2relation = {}, {}
        drug_set = json.load(open(os.path.join(self.task_dir, "data/node2id.json")))
        entity_set = json.load(open(os.path.join(self.task_dir, "data/entity_drug.json")))
        relation_set = json.load(open(os.path.join(self.task_dir, "data/relation2id.json")))

        for drug in drug_set:
            id2entity[int(drug_set[drug])] = drug
        for ent in entity_set:
            id2entity[int(entity_set[ent])] = ent
        for rel in relation_set:
            id2relation[int(rel)] = relation_set[rel]

        self.id2entity = id2entity
        self.id2relation = id2relation

    # ------------------------------------------------------------------
    def load_graph(self, triplets: np.ndarray):
        """把三元组集合转换为稀疏张量  [E,E,2R+1]"""
        edges = self.double_triple(triplets)
        # 单位边 (u,u,idd)  ─ idd = 2R
        idd = np.concatenate(
            [
                np.expand_dims(np.arange(self.all_ent), 1),
                np.expand_dims(np.arange(self.all_ent), 1),
                2 * self.all_rel * np.ones((self.all_ent, 1)),
            ],
            1,
        )
        edges = np.concatenate([edges, idd], axis=0)
        values = np.ones(edges.shape[0])

        adj = torch.sparse_coo_tensor(
            indices=torch.LongTensor(edges).t(),
            values=torch.FloatTensor(values),
            size=torch.Size([self.all_ent, self.all_ent, 2 * self.all_rel + 1]),
            requires_grad=False,
        )
        return safe_cuda(adj)

    # ------------------------------------------------------------------
    @staticmethod
    def double_triple(triplet):
        """(h,t,r) ↦ {(t,h,r),(h,t,r+R)}"""
        new_triples, n_rel = [], None
        if len(triplet):
            n_rel = triplet[:, 2].max() + 1
        for h, t, r in triplet:
            new_triples.append([t, h, r])
            new_triples.append([h, t, r + n_rel])
        return np.array(new_triples, dtype="int")


# ======================================================================
#                       单子图 DataLoaderSubKG
# ======================================================================
class DataLoaderSubKG(DataLoader):
    def __init__(self, params, kg_suffix: str):
        self.kg_suffix = kg_suffix
        super().__init__(params)

        kg_paths = {
            "train": os.path.join(self.task_dir, f"data/{self.dataset}/train_{kg_suffix}_KG.txt"),
            "valid": os.path.join(self.task_dir, f"data/{self.dataset}/valid_{kg_suffix}_KG.txt"),
            "test":  os.path.join(self.task_dir, f"data/{self.dataset}/test_{kg_suffix}_KG.txt"),
        }
        self.process_files_kg(kg_paths)
        self.prepare_for_training()

    # ------------------------------------------------------------------
    def process_files_kg(self, paths: Dict[str, str]):
        self.kg_triplets = defaultdict(list)
        self.ddi_in_kg = set()

        for split, file_path in paths.items():
            with open(file_path) as f:
                lines = [ln.split() for ln in f.read().split("\n") if ln.strip()]

            for h, t, r in lines:
                h, t, r = int(h), int(t), int(r)
                self.entity2id.setdefault(h, h)
                self.entity2id.setdefault(t, t)
                self.relation2id.setdefault(r, r)
                self.kg_triplets[split].append([h, t, r])

                if split == "train":
                    if h in self.train_ent:
                        self.ddi_in_kg.add(h)
                    if t in self.train_ent:
                        self.ddi_in_kg.add(t)

        # ndarray 转换
        self.train_kg = np.array(self.kg_triplets["train"], dtype="int")
        self.valid_kg = np.array(
            self.kg_triplets["train"] + self.kg_triplets["valid"], dtype="int"
        )
        self.test_kg = np.array(
            self.kg_triplets["train"] + self.kg_triplets["valid"] + self.kg_triplets["test"],
            dtype="int",
        )

        self.all_ent = max(self.entity2id.keys()) + 1
        self.all_rel = max(self.relation2id.keys()) + 1

        # 打印 KG & DDI 统计
        print(
            f"KG triplets ({self.kg_suffix}): "
            f"Train-{len(self.train_kg)} "
            f"Valid-{len(self.valid_kg)} "
            f"Test-{len(self.test_kg)}"
        )

    # ------------------------------------------------------------------
    def prepare_for_training(self, ratio=0.8):
        """根据 S0/S1/S2 场景生成训练图 & 训练样本"""
        all_triplet = np.array(self.triplets["train"])
        
        # 训练图使用ratio=0.8
        n_ent_kg = len(self.ddi_in_kg)
        train_ent = set(self.train_ent) - set(
            np.random.choice(list(self.ddi_in_kg), n_ent_kg - int(n_ent_kg * ratio))
        )

        # ---- 根据数据集场景拆分 ----
        fact_triplet, train_data = [], []
        if self.dataset.startswith("S1"):
            for h, t, r in all_triplet:
                if h in train_ent and t in train_ent:
                    fact_triplet.append([h, t, r])
                elif h in train_ent or t in train_ent:
                    train_data.append([h, t, r])

        elif self.dataset.startswith("S2"):
            for h, t, r in all_triplet:
                if h in train_ent and t in train_ent:
                    fact_triplet.append([h, t, r])
                elif h not in train_ent and t not in train_ent:
                    train_data.append([h, t, r])

        else:  # S0 或其他：随机 80% 构图
            n_all = len(all_triplet)
            rand_idx = np.random.permutation(n_all)
            all_triplet = all_triplet[rand_idx]
            n_fact = int(n_all * 0.8)
            fact_triplet = all_triplet[:n_fact].tolist()
            train_data = all_triplet[n_fact:].tolist()

        fact_triplet = np.array(fact_triplet, dtype="int")
        self.train_data = np.array(train_data, dtype="int")
        self.n_train = len(self.train_data)

        # 训练图 = DDI fact + 子 KG(train)
        kg_triplets = np.concatenate([fact_triplet, self.train_kg], axis=0)
        self.KG = self.load_graph(kg_triplets)

        # 验证和测试时使用完整图(ratio=1)
        # 为验证和测试重新构建fact_triplet
        full_train_ent = self.train_ent  # 使用完整训练实体集
        full_fact_triplet = []
        
        if self.dataset.startswith("S1"):
            for h, t, r in all_triplet:
                if h in full_train_ent and t in full_train_ent:
                    full_fact_triplet.append([h, t, r])
        elif self.dataset.startswith("S2"):
            for h, t, r in all_triplet:
                if h in full_train_ent and t in full_train_ent:
                    full_fact_triplet.append([h, t, r])
        else:  # S0或其他情况
            n_all = len(all_triplet)
            # 使用全部训练三元组作为fact
            full_fact_triplet = all_triplet.tolist()
            
        full_fact_triplet = np.array(full_fact_triplet, dtype="int")
        
        # 验证、测试图使用完整的实体集
        self.vKG = self.load_graph(np.concatenate([self.triplets['train'], self.valid_kg], axis=0))
        self.tKG = self.load_graph(np.concatenate([self.triplets['train'], self.triplets['valid'], self.test_kg], axis=0))

        print(f"Training data size: {self.n_train}")


# ======================================================================
#                       多专家 MoEDataLoader（原逻辑保持）
# ======================================================================
class MoEDataLoader(DataLoader):
    """为多专家模型加载 6 个子 KG"""

    def __init__(self, params, kg_names: List[str] = None):
        if kg_names is None:
            kg_names = ["Gene_Disease", "molecule", "PCiC", "CcSE", "CrC", "fused"]
        self.kg_names = list(kg_names)
        super().__init__(params)  # 处理 DDI

        self.kg_triplets_multi = {name: defaultdict(list) for name in self.kg_names}
        self.ddi_in_kg = set()
        self.load_multi_kg()
        self.prepare_for_training()

    # ------------------------------------------------------------------
    def load_multi_kg(self):
        for name in self.kg_names:
            for split in ["train", "valid", "test"]:
                fp = os.path.join(self.task_dir, f"data/{self.dataset}/{split}_{name}_KG.txt")
                with open(fp) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        h, t, r = map(int, line.split())
                        self.entity2id.setdefault(h, h)
                        self.entity2id.setdefault(t, t)
                        self.relation2id.setdefault(r, r)
                        self.kg_triplets_multi[name][split].append([h, t, r])
                        if split == "train":
                            if h in self.train_ent:
                                self.ddi_in_kg.add(h)
                            if t in self.train_ent:
                                self.ddi_in_kg.add(t)

        self.all_ent = max(self.entity2id.keys()) + 1
        self.all_rel = max(self.relation2id.keys()) + 1

        # 打印统计
        for name in self.kg_names:
            train_kg = len(self.kg_triplets_multi[name]["train"])
            valid_kg = train_kg + len(self.kg_triplets_multi[name]["valid"])
            test_kg = valid_kg + len(self.kg_triplets_multi[name]["test"])
            print(f"KG triplets ({name}): Train-{train_kg} Valid-{valid_kg} Test-{test_kg}")

    # ------------------------------------------------------------------
    def _to_array(self, lst):
        return np.array(lst, dtype="int") if len(lst) else np.zeros((0, 3), dtype="int")

    def prepare_for_training(self, ratio=0.8):
        n_ent_kg = len(self.ddi_in_kg)
        train_ent = set(self.train_ent) - set(
            np.random.choice(list(self.ddi_in_kg), n_ent_kg - int(n_ent_kg * ratio))
        )
        all_triplet = np.array(self.triplets["train"])

        fact_triplet, train_data = [], []
        if self.dataset.startswith("S1"):
            for h, t, r in all_triplet:
                if h in train_ent and t in train_ent:
                    fact_triplet.append([h, t, r])
                elif h in train_ent or t in train_ent:
                    train_data.append([h, t, r])
        elif self.dataset.startswith("S2"):
            for h, t, r in all_triplet:
                if h in train_ent and t in train_ent:
                    fact_triplet.append([h, t, r])
                elif h not in train_ent and t not in train_ent:
                    train_data.append([h, t, r])
        else:
            n_all = len(all_triplet)
            rand_idx = np.random.permutation(n_all)
            n_fact = int(n_all * 0.8)
            fact_triplet = all_triplet[rand_idx][:n_fact].tolist()
            train_data = all_triplet[rand_idx][n_fact:].tolist()

        fact_triplet = np.array(fact_triplet, dtype="int")
        self.train_data = np.array(train_data, dtype="int")
        self.n_train = len(self.train_data)

        # 全局 DDI 训练图（仅 fact_triplet）
        self.KG = self.load_graph(fact_triplet)
        
        # 为验证和测试重新构建完整的fact_triplet（ratio=1）
        full_train_ent = self.train_ent  # 使用完整训练实体集
        full_fact_triplet = []
        
        if self.dataset.startswith("S1"):
            for h, t, r in all_triplet:
                if h in full_train_ent and t in full_train_ent:
                    full_fact_triplet.append([h, t, r])
        elif self.dataset.startswith("S2"):
            for h, t, r in all_triplet:
                if h in full_train_ent and t in full_train_ent:
                    full_fact_triplet.append([h, t, r])
        else:
            # 使用全部训练三元组作为fact
            full_fact_triplet = all_triplet.tolist()
            
        full_fact_triplet = np.array(full_fact_triplet, dtype="int")

        # 为每个专家构建 KG 字典
        self.KG_dict, self.vKG_dict, self.tKG_dict = {}, {}, {}
        for name in self.kg_names:
            train_arr = self._to_array(self.kg_triplets_multi[name]["train"])
            valid_arr = self._to_array(self.kg_triplets_multi[name]["valid"])
            test_arr = self._to_array(self.kg_triplets_multi[name]["test"])

            # 训练图使用 ratio=0.8 的 fact_triplet
            self.KG_dict[name] = self.load_graph(np.concatenate([fact_triplet, train_arr], 0))
            
            # 验证图：直接使用所有训练三元组与验证集合并
            self.vKG_dict[name] = self.load_graph(
                np.concatenate([self.triplets['train'], 
                               self._to_array(self.kg_triplets_multi[name]["train"] + 
                                             self.kg_triplets_multi[name]["valid"])], 0)
            )
            
            # 测试图：直接使用所有训练三元组与验证集、测试集合并
            self.tKG_dict[name] = self.load_graph(
                np.concatenate([self.triplets['train'], self.triplets['valid'],
                               self._to_array(self.kg_triplets_multi[name]["train"] + 
                                             self.kg_triplets_multi[name]["valid"] + 
                                             self.kg_triplets_multi[name]["test"])], 0)
            )

        print(f"Training data size: {self.n_train}")
