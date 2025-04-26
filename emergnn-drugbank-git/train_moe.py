import argparse
import torch
import numpy as np
import random

from combined_loader import MoEDataLoader
from combined_base_model import MoEBaseModel


def apply_dataset_defaults(args):
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
    return args


def train_moe(args):
    # 设置随机种子，但每次运行使用不同的种子
    seed = args.seed if args.seed is not None else random.randint(1, 10000)
    print(f"Using random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dl = MoEDataLoader(args)
    args.all_ent, args.all_rel, args.eval_rel = dl.all_ent, dl.all_rel, dl.eval_rel
    ddi_graph = dl.KG

    ckpts = args.expert_ckpts.split(",") if args.expert_ckpts else None
    model = MoEBaseModel(dl.kg_names, dl.eval_ent, dl.eval_rel, args, expert_ckpts=ckpts)

    best_f1 = -1
    for epoch in range(args.n_epoch):
        dl.prepare_for_training()  # ✅ 每轮构图 + 重新拆分训练数据
        train_pos = torch.LongTensor(dl.train_data)
        if torch.cuda.is_available():
            train_pos = train_pos.cuda()

        # 训练，并返回平均损失
        train_loss = model.train(train_pos, dl.KG_dict, ddi_graph)
        print(f"[E{epoch}] loss={train_loss:.4f}", end=" ")

        # 每个epoch结束后评估valid和test
        valid_pos = torch.LongTensor(dl.triplets["valid"])
        test_pos = torch.LongTensor(dl.triplets["test"])
        if torch.cuda.is_available():
            valid_pos, test_pos = valid_pos.cuda(), test_pos.cuda()

        v_f1, v_acc, v_kap = model.evaluate(valid_pos, dl.vKG_dict, ddi_graph)
        t_f1, t_acc, t_kap = model.evaluate(test_pos, dl.tKG_dict, ddi_graph)
        print(f"| [Epoch {epoch}] valid F1={v_f1:.4f} acc={v_acc:.4f} κ={v_kap:.4f} test F1={t_f1:.4f} acc={t_acc:.4f} κ={t_kap:.4f}")
        
        # 记录指标到日志
        model.log_metrics(epoch, v_f1, v_acc, v_kap, log_type="valid")
        model.log_metrics(epoch, t_f1, t_acc, t_kap, log_type="test")
        
        # 更新学习率调度器
        model.update_scheduler(v_f1)

        if v_f1 > best_f1:
            best_f1 = v_f1
            if args.save_model:
                # 使用新的检查点命名规则
                ckpt = model.get_best_checkpoint_path()
                model.save_model(ckpt)
                print(">> saved:", ckpt)


def evaluate_moe(args, ckpt):
    dl = MoEDataLoader(args)
    args.all_ent, args.all_rel, args.eval_rel = dl.all_ent, dl.all_rel, dl.eval_rel
    ddi_graph = dl.KG

    model = MoEBaseModel(dl.kg_names, dl.eval_ent, dl.eval_rel, args)
    model.model.load_state_dict(torch.load(ckpt))

    valid_pos = torch.LongTensor(dl.triplets["valid"])
    test_pos = torch.LongTensor(dl.triplets["test"])
    if torch.cuda.is_available():
        valid_pos, test_pos = valid_pos.cuda(), test_pos.cuda()

    v_f1, v_acc, v_kap = model.evaluate(valid_pos, dl.vKG_dict, ddi_graph)
    t_f1, t_acc, t_kap = model.evaluate(test_pos, dl.tKG_dict, ddi_graph)
    print(">>", ckpt)
    print(f"Valid F1={v_f1:.4f} acc={v_acc:.4f} κ={v_kap:.4f}")
    print(f"Test  F1={t_f1:.4f} acc={t_acc:.4f} κ={t_kap:.4f}")
    
    # 记录测试结果到日志
    model.log_metrics(-1, v_f1, v_acc, v_kap, log_type="valid_eval")
    model.log_metrics(-1, t_f1, t_acc, t_kap, log_type="test_eval")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task_dir", default="./")
    p.add_argument("--dataset", default="S1_1")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--expert_ckpts", type=str, default="")
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--lamb", type=float, default=7e-4)
    p.add_argument("--n_epoch", type=int, default=100)
    p.add_argument("--n_batch", type=int, default=16)
    p.add_argument("--n_dim", type=int, default=128)
    p.add_argument("--length", type=int, default=3)
    p.add_argument("--feat", type=str, default="M", choices=["E", "M"])
    p.add_argument("--test_batch_size", type=int, default=8)   
    p.add_argument("--epoch_per_test", type=int, default=1)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--load_model", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--gate_hidden", type=int, default=128)
    p.add_argument("--gate_type", type=str, default="soft", choices=["soft", "topk"])
    p.add_argument("--gate_topk", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gate_lamb", type=float, default=0.01,
               help="门控熵正则系数，0 可关闭")

    args = p.parse_args()
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    # ---- 覆盖默认超参 ----
    args = apply_dataset_defaults(args)

    # ---- 打印最终超参 ----
    print("========== 模型参数 (args) =========")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("===================================")

    if args.eval_only and args.model_path:
        evaluate_moe(args, args.model_path)
    else:
        train_moe(args)
