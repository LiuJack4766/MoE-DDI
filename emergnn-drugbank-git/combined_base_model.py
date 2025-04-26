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
                f.write("\n时间戳,Epoch,类型,F1,Acc,Kappa\n")
        with open(log_file, "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts},{epoch},{log_type},{f1:.4f},{acc:.4f},{kap:.4f}\n")

    def get_best_checkpoint_path(self):
        return os.path.join(RESULT_DIR,
                            f"{self.args.kg_suffix}_bestckpoint_{self.args.length}.pt")

# =============== 多专家 / MoE ===============
class MoEBaseModel:
    def __init__(self, kg_names, eval_ent, eval_rel, args, expert_ckpts=None):
        from combined_models import MoE_EmerGNN
        self.model = MoE_EmerGNN(kg_names, eval_ent, eval_rel, args)
        if torch.cuda.is_available():
            self.model.cuda()
        self.kg_names = kg_names
        self.args = args
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
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
                f.write("\n时间戳,Epoch,类型,F1,Acc,Kappa\n")
        with open(log_file, "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts},{epoch},{log_type},{f1:.4f},{acc:.4f},{kap:.4f}\n")

    def get_best_checkpoint_path(self):
        names = [f"{n}_{self.args.length}" for n in self.kg_names]
        return os.path.join(RESULT_DIR, "_".join(names) + "_bestcheckpoint.pt")
