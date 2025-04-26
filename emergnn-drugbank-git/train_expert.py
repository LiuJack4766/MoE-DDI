#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单专家训练脚本（含断点续训 + 数据集专用默认超参）
2025-04-25
"""
import argparse, random, numpy as np, torch, os, sys
from combined_loader import DataLoaderSubKG
from combined_base_model import BaseModel


# ---------- 数据集专用超参 ---------
def apply_dataset_defaults(args):
    """根据数据集前缀 S0 / S1 / S2 自动覆盖一组最优超参"""
    ds = args.dataset.lower()
    if ds.startswith("s1") or ds.startswith("s2"):
        args.lr = 0.001
        args.lamb = 1e-8
        args.n_batch = 32
        args.n_dim = 64
        args.length = 3
        args.feat = "M"
    elif ds.startswith("s0"):
        args.lr = 0.001
        args.lamb = 1e-6
        args.n_batch = 128
        args.n_dim = 64
        args.length = 3
        args.feat = "E"
    # 其余数据集保持命令行参数
    return args


# ---------- 工具 ----------
def set_seed(seed):
    print(f"Using random seed: {seed}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ---------- 训练 ----------
def train_expert(args):
    set_seed(args.seed if args.seed else random.randint(1, 10000))

    # 数据
    dl = DataLoaderSubKG(args, args.kg_suffix)
    args.all_ent, args.all_rel, args.eval_rel = dl.all_ent, dl.all_rel, dl.eval_rel
    model = BaseModel(dl.eval_ent, dl.eval_rel, args)

    # —— 可选：加载 checkpoint 继续训练 ——
    best_f1 = -1
    if args.load_model:
        ckpt = args.model_path or model.get_best_checkpoint_path()
        if os.path.isfile(ckpt):
            model.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f">> loaded checkpoint: {ckpt}")
            val_pos = torch.LongTensor(dl.triplets["valid"])
            if torch.cuda.is_available(): val_pos = val_pos.cuda()
            best_f1, acc, kap = model.evaluate(val_pos, None, dl.vKG)
            print(f">> checkpoint valid F1={best_f1:.4f} acc={acc:.4f} κ={kap:.4f}")
        else:
            print(f">> checkpoint not found: {ckpt}")
    dl.prepare_for_training()  #  每轮构图 + 重新拆分训练数据

    # 训练循环
    for epoch in range(args.n_epoch):
        dl.prepare_for_training()  #  每轮构图 + 重新拆分训练数据
        train_pos = torch.LongTensor(dl.train_data)
        if torch.cuda.is_available(): train_pos = train_pos.cuda()

        loss = model.train(train_pos, None, dl.KG)
        print(f"[E{epoch}] loss={loss:.4f}", end=" ")

        # 每个epoch结束后评估valid和test
        val_pos = torch.LongTensor(dl.triplets["valid"])
        test_pos = torch.LongTensor(dl.triplets["test"])
        if torch.cuda.is_available(): 
            val_pos = val_pos.cuda()
            test_pos = test_pos.cuda()
            
        v_f1, v_acc, v_kap = model.evaluate(val_pos, None, dl.vKG)
        t_f1, t_acc, t_kap = model.evaluate(test_pos, None, dl.tKG)
        print(f"| [Epoch {epoch}] valid F1={v_f1:.4f} acc={v_acc:.4f} κ={v_kap:.4f} test F1={t_f1:.4f} acc={t_acc:.4f} κ={t_kap:.4f}")

        # 记录指标
        model.log_metrics(epoch, v_f1, v_acc, v_kap, "valid")
        model.log_metrics(epoch, t_f1, t_acc, t_kap, "test")
        model.update_scheduler(v_f1)

        if v_f1 > best_f1:
            best_f1 = v_f1
            if args.save_model:
                ckpt = model.get_best_checkpoint_path()
                model.save_model(ckpt)
                print(">> saved:", ckpt)


# ---------- 评估 ----------
def evaluate_expert(args, ckpt):
    dl = DataLoaderSubKG(args, args.kg_suffix)
    args.all_ent, args.all_rel, args.eval_rel = dl.all_ent, dl.all_rel, dl.eval_rel
    model = BaseModel(dl.eval_ent, dl.eval_rel, args)
    model.model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    val_pos = torch.LongTensor(dl.triplets["valid"])
    test_pos = torch.LongTensor(dl.triplets["test"])
    if torch.cuda.is_available(): val_pos, test_pos = val_pos.cuda(), test_pos.cuda()

    v = model.evaluate(val_pos, None, dl.vKG)
    t = model.evaluate(test_pos, None, dl.tKG)
    print(">>", ckpt)
    print(f"Valid F1={v[0]:.4f} acc={v[1]:.4f} κ={v[2]:.4f}")
    print(f"Test  F1={t[0]:.4f} acc={t[1]:.4f} κ={t[2]:.4f}")
    model.log_metrics(-1, *v, "valid_eval")
    model.log_metrics(-1, *t, "test_eval")


# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task_dir", default="./")
    p.add_argument("--dataset",  default="S1_1")
    p.add_argument("--kg_suffix", required=True)

    # 通用初始值（可能被自动覆盖）
    p.add_argument("--lr",   type=float, default=0.03)
    p.add_argument("--lamb", type=float, default=7e-4)
    p.add_argument("--n_epoch", type=int, default=100)
    p.add_argument("--n_batch", type=int, default=96)
    p.add_argument("--n_dim",  type=int, default=128)
    p.add_argument("--length", type=int, default=3)
    p.add_argument("--feat",   type=str, default="M", choices=["E", "M"])
    p.add_argument("--test_batch_size", type=int, default=16)
    p.add_argument("--epoch_per_test",  type=int, default=1)

    # 训练 / 评估配置
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--load_model", action="store_true")
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--gpu",  type=int, default=0)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--seed", type=int, default=1234)

    args = p.parse_args()
    if torch.cuda.is_available() and args.gpu >= 0: torch.cuda.set_device(args.gpu)

    # ---- 覆盖默认超参 ----
    args = apply_dataset_defaults(args)

    # ---- 打印最终超参 ----
    print("========== 模型参数 (args) =========")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("===================================")

    if args.eval_only and args.model_path:
        evaluate_expert(args, args.model_path)
    else:
        train_expert(args)