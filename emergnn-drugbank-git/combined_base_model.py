# combined_base_model.py
import os, datetime, numpy as np, torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, cohen_kappa_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import batch_by_size

RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

def safe_cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

# =============== 单专家 ===============
class BaseModel:
    def __init__(self, eval_ent, eval_rel, args):
        from combined_models import EmerGNN
        self.model = EmerGNN(eval_ent, eval_rel, args)
        if torch.cuda.is_available():
            self.model.cuda()

        self.args = args
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max")

    @staticmethod
    def _emergnn_loss(logits, labels):
        p_score = logits[torch.arange(len(labels)), labels]
        max_n   = torch.max(logits, 1, keepdim=True)[0]
        loss_v  = -p_score + max_n.squeeze() + torch.log(torch.exp(logits - max_n).sum(1))
        return loss_v.sum()

    def train(self, train_pos, _unused, KG):
        h, t, r = train_pos[:,0], train_pos[:,1], train_pos[:,2]
        n_batch = self.args.n_batch
        self.model.train()

        total_iter = len(h) // n_batch + int(len(h)%n_batch>0)
        loss_sum = 0.0
        for bh, bt, br in tqdm(batch_by_size(n_batch, h, t, r),
                               total=total_iter, desc="Train", ncols=100, leave=False):
            self.optimizer.zero_grad()
            logits = self.model.enc_r(self.model.enc_ht(bh, bt, KG))
            loss = self._emergnn_loss(logits, br)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
        # —— 可选：保持 scheduler.step_count 同步 —— 
        #self.scheduler._step_count += 1
        return loss_sum / total_iter

    def evaluate(self, data_pos, _unused, KG):
        h, t, r = data_pos[:,0], data_pos[:,1], data_pos[:,2]
        bs = self.args.test_batch_size
        self.model.eval()

        probs, labels = [], []
        total_iter = len(h)//bs + int(len(h)%bs>0)
        with torch.no_grad():
            for bh, bt, br in tqdm(batch_by_size(bs, h, t, r),
                                   total=total_iter, desc="Eval ", ncols=100, leave=False):
                logits = self.model.enc_r(self.model.enc_ht(bh, bt, KG))
                probs.append(F.softmax(logits, -1).cpu())
                labels.append(br.cpu())

        y_true = torch.cat(labels).numpy()
        y_pred = torch.cat(probs).numpy().argmax(1)
        acc = (y_pred == y_true).mean()
        f1  = f1_score(y_true, y_pred, average="macro")
        kap = 0.0 if len(np.unique(y_true))<=1 or len(np.unique(y_pred))<=1 \
              else cohen_kappa_score(y_true, y_pred)
        return f1, acc, kap

    def update_scheduler(self, metric):
        self.scheduler.step(metric)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def log_metrics(self, epoch, f1, acc, kap, log_type="valid"):
        log_file = os.path.join(RESULT_DIR, f"{self.args.kg_suffix}_log.txt")
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"==== 训练开始: {ts} ====\n超参数:\n")
                for k, v in vars(self.args).items(): f.write(f"  {k}: {v}\n")
                
                # 记录特征矩阵维度信息到日志文件
                f.write("\n特征矩阵维度信息:\n")
                if self.args.feat.upper() == "M":
                    f.write(f"  - 输入维度: 1024 (Morgan指纹)\n")
                    f.write(f"  - 实体嵌入维度: {self.args.n_dim}\n")
                    f.write(f"  - 关系嵌入维度: {self.args.n_dim}\n")
                    f.write(f"  - 注意力权重维度: {2*self.args.all_rel+1}\n")
                    f.write(f"  - 最终预测维度: {self.model.eval_rel}\n")
                    f.write(f"  - 连接方式: head_hid + tail_hid → {2*self.args.n_dim} → {self.model.eval_rel}\n")
                else:
                    f.write(f"  - 实体嵌入维度: {self.args.n_dim}\n")
                    f.write(f"  - 关系嵌入维度: {self.args.n_dim}\n")
                    f.write(f"  - 注意力权重维度: {2*self.args.all_rel+1}\n")
                    f.write(f"  - 最终预测维度: {self.model.eval_rel}\n")
                    f.write(f"  - 连接方式: head_emb + tail_emb + head_hid + tail_hid → {4*self.args.n_dim} → {self.model.eval_rel}\n")
                
                f.write("\n时间戳,Epoch,类型,F1,Acc,Kappa\n")
        with open(log_file, "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts},{epoch},{log_type},{f1:.4f},{acc:.4f},{kap:.4f}\n")

    def get_best_checkpoint_path(self):
        return os.path.join(RESULT_DIR,
                            f"{self.args.kg_suffix}_bestckpoint_{self.args.length}.pt")

# =============== 多专家 / MoE ===============
class MoEBaseModel:
    def __init__(self, kg_names, eval_ent, eval_rel, args, expert_ckpts=None, freeze_experts=False):
        from combined_models import MoE_EmerGNN
        self.model = MoE_EmerGNN(kg_names, eval_ent, eval_rel, args)
        if torch.cuda.is_available():
            self.model.cuda()
        self.kg_names = kg_names
        self.args = args
        
        # 加载预训练专家或完整MoE模型
        if args.model_path and os.path.exists(args.model_path):
            print(f">> 加载完整MoE模型：{args.model_path}")
            
            # 如果没有明确指定专家名称，尝试从检查点路径中推断
            if not kg_names or len(kg_names) == 0:
                # 解析类似 "fused_3_molecule_3_bestcheckpoint.pt" 的格式
                base_name = os.path.basename(args.model_path).split('_bestcheckpoint')[0]
                parts = base_name.split('_')
                
                # 跳过数字部分来获取专家名称
                inferred_kg_names = []
                i = 0
                while i < len(parts):
                    if not parts[i].isdigit():  # 当前部分是专家名称
                        inferred_kg_names.append(parts[i])
                        i += 2  # 跳过专家名称后面的数字参数
                    else:
                        i += 1  # 跳过单独的数字
                
                if inferred_kg_names:
                    print(f">> 从检查点路径推断专家名称: {inferred_kg_names}")
                    # 更新模型的kg_names
                    self.kg_names = inferred_kg_names
                    # 重新创建MoE_EmerGNN模型
                    self.model = MoE_EmerGNN(inferred_kg_names, eval_ent, eval_rel, args)
                    if torch.cuda.is_available():
                        self.model.cuda()
            
            self.model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        elif expert_ckpts:
            # 分别加载每个专家模型
            for i, (name, ckpt) in enumerate(zip(kg_names, expert_ckpts)):
                if os.path.exists(ckpt):
                    print(f">> 加载专家 {name} 参数：{ckpt}")
                    self.model.load_expert(name, ckpt)
        
        # 冻结专家参数（只训练门控网络）
        if freeze_experts:
            print(">> 冻结专家参数，只训练门控网络")
            for name, param in self.model.named_parameters():
                if not name.startswith('gate'):
                    param.requires_grad = False
        
        # 只优化需要梯度的参数
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                             lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max")

    def _moe_loss(self, logits, labels, heads, tails):
        base = BaseModel._emergnn_loss(logits, labels)
        plain_ht = self.model._get_plain_ht(heads, tails)
        gate_logits = self.model.gate(plain_ht)
        if self.model.gate_type == "soft":
            alpha = torch.softmax(gate_logits, -1)
        else:
            idx   = torch.topk(gate_logits, self.model.top_k, -1).indices
            alpha = torch.zeros_like(gate_logits)
            alpha.scatter_(1, idx, 1.0/self.model.top_k)
        entropy = (-alpha * torch.log(alpha + 1e-12)).sum(-1).mean()
        return base + self.args.gate_lamb * entropy

    def train(self, train_pos, kg_dict, ddi_graph):
        h, t, r = train_pos[:,0], train_pos[:,1], train_pos[:,2]
        n_batch = self.args.n_batch
        self.model.train()

        total_iter = len(h)//n_batch + int(len(h)%n_batch>0)
        loss_sum = 0.0
        for bh, bt, br in tqdm(batch_by_size(n_batch, h, t, r),
                               total=total_iter, desc="Train", ncols=100, leave=False):
            self.optimizer.zero_grad()
            logits = self.model(bh, bt, ddi_graph, kg_dict)
            loss   = self._moe_loss(logits, br, bh, bt)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()
        return loss_sum / total_iter

    def evaluate(self, data_pos, kg_dict, ddi_graph):
        h, t, r = data_pos[:,0], data_pos[:,1], data_pos[:,2]
        bs = self.args.test_batch_size
        self.model.eval()

        probs, labels = [], []
        total_iter = len(h)//bs + int(len(h)%bs>0)
        with torch.no_grad():
            for bh, bt, br in tqdm(batch_by_size(bs, h, t, r),
                                   total=total_iter, desc="Eval ", ncols=100, leave=False):
                logits = self.model(bh, bt, ddi_graph, kg_dict)
                probs.append(F.softmax(logits, -1).cpu())
                labels.append(br.cpu())

        y_true = torch.cat(labels).numpy()
        y_pred = torch.cat(probs).numpy().argmax(1)
        acc = (y_pred==y_true).mean()
        f1  = f1_score(y_true, y_pred, average="macro")
        kap = 0.0 if len(np.unique(y_true))<=1 or len(np.unique(y_pred))<=1 \
              else cohen_kappa_score(y_true, y_pred)
        return f1, acc, kap

    def update_scheduler(self, metric):
        self.scheduler.step(metric)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def log_metrics(self, epoch, f1, acc, kap, log_type="valid"):
        log_file = os.path.join(RESULT_DIR, "_".join(self.kg_names) + "_log.txt")
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"==== 训练开始: {ts} ====\n超参数:\n")
                for k,v in vars(self.args).items(): f.write(f"  {k}: {v}\n")
                
                # 记录特征矩阵维度信息到日志文件
                n_experts = len(self.kg_names)
                f.write("\n特征矩阵维度信息:\n")
                if self.args.feat.upper() == "M":
                    f.write(f"  - 输入维度: 1024 (Morgan指纹)\n")
                    f.write(f"  - 实体嵌入维度: {self.args.n_dim}\n")
                    f.write(f"  - 关系嵌入维度: {self.args.n_dim}\n")
                    f.write(f"  - 专家数量: {n_experts}\n")
                    f.write(f"  - 注意力权重维度: {2*self.args.all_rel+1}\n")
                    f.write(f"  - 最终预测维度: {self.model.eval_rel}\n")
                    f.write(f"  - 专家结构: head_hid + tail_hid → {2*self.args.n_dim} → {self.model.eval_rel}\n")
                else:
                    f.write(f"  - 实体嵌入维度: {self.args.n_dim}\n")
                    f.write(f"  - 关系嵌入维度: {self.args.n_dim}\n")
                    f.write(f"  - 专家数量: {n_experts}\n")
                    f.write(f"  - 注意力权重维度: {2*self.args.all_rel+1}\n")
                    f.write(f"  - 最终预测维度: {self.model.eval_rel}\n")
                    f.write(f"  - 专家结构: head_emb + tail_emb + head_hid + tail_hid → {4*self.args.n_dim} → {self.model.eval_rel}\n")
                
                # 记录门控网络信息
                f.write("\n门控网络维度:\n")
                f.write(f"  - 输入维度: {2*self.args.n_dim}\n")
                if self.args.gate_hidden > 0:
                    f.write(f"  - 隐藏层维度: {self.args.gate_hidden}\n")
                f.write(f"  - 输出维度: {n_experts}\n")
                f.write(f"  - 门控类型: {self.args.gate_type}" + (f" (top-k={self.args.gate_topk})" if self.args.gate_type == "topk" else "") + "\n")
                f.write(f"  - 熵正则系数: {self.args.gate_lamb}\n")
                
                f.write("\n时间戳,Epoch,类型,F1,Acc,Kappa\n")
        with open(log_file, "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts},{epoch},{log_type},{f1:.4f},{acc:.4f},{kap:.4f}\n")

    def get_best_checkpoint_path(self):
        names = [f"{n}_{self.args.length}" for n in self.kg_names]
        return os.path.join(RESULT_DIR, "_".join(names) + "_bestcheckpoint.pt")
