# tune_hyperms_moe.py
import os
import argparse
import random
import torch
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial

from combined_loader import MoEDataLoader
from combined_base_model import MoEBaseModel

# 解析命令行参数
parser = argparse.ArgumentParser(description="Hyper-parameter tuning for MoE EmerGNN")
parser.add_argument('--task_dir', type=str, default='./', help='root path of datasets')
parser.add_argument('--dataset', type=str, default='S1_1', help='dataset sub-folder')
parser.add_argument('--lamb', type=float, default=7e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='GPU id')
parser.add_argument('--n_dim', type=int, default=128, help='embedding dim')
parser.add_argument('--lr', type=float, default=3e-3, help='initial learning rate')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--load_model', action='store_true')
parser.add_argument('--n_epoch', type=int, default=100, help='training epochs')
parser.add_argument('--n_batch', type=int, default=512, help='batch size')
parser.add_argument('--epoch_per_test', type=int, default=10, help='test frequency')
parser.add_argument('--test_batch_size', type=int, default=8, help='test batch size')
parser.add_argument('--seed', type=int, default=1234)

# 门控网络超参
parser.add_argument('--gate_hidden', type=int, default=128, help='hidden width of gate; 0 = no hidden layer')
parser.add_argument('--gate_type', default='soft', choices=['soft', 'topk'], help='gate strategy: soft / topk')
parser.add_argument('--gate_topk', type=int, default=2, help='k for top-k gate')

# 加载专家检查点路径
parser.add_argument('--expert_ckpts', type=str, default='',
                    help="Path to pre-trained expert model checkpoints (comma-separated)")

args = parser.parse_args()

# 设置随机种子和设备
if torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise RuntimeError(f'GPU id {args.gpu} out of range.')
    torch.cuda.set_device(args.gpu)
else:
    args.gpu = -1
os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# 加载数据
dataloader = MoEDataLoader(args)
eval_ent, eval_rel = dataloader.eval_ent, dataloader.eval_rel
args.all_ent, args.all_rel, args.eval_rel = (dataloader.all_ent, dataloader.all_rel, eval_rel)

# 准备评估数据
train_pos = torch.LongTensor(dataloader.train_data)
valid_pos = torch.LongTensor(dataloader.triplets['valid'])
test_pos = torch.LongTensor(dataloader.triplets['test'])
if torch.cuda.is_available():
    train_pos = train_pos.cuda()
    valid_pos = valid_pos.cuda()
    test_pos = test_pos.cuda()

# DDI图作为基础图
ddi_graph = dataloader.KG

# 超参数搜索空间
space = {
    # 原有超参
    "lr": hp.choice("lr", [1e-2, 3e-3, 1e-3, 3e-4]),
    "lamb": hp.choice("lamb", [1e-8, 1e-6, 1e-4, 1e-2]),
    "n_batch": hp.choice("n_batch", [32, 64, 128]),
    "n_dim": hp.choice("n_dim", [32, 64, 128]),
    "length": hp.choice("length", [2, 3, 4, 5]),
    "feat": hp.choice("feat", ['M', 'E']),
    # 门控网络超参
    "gate_hidden": hp.choice("gate_hidden", [0, 64, 128, 256]),
    "gate_type": hp.choice("gate_type", ['soft', 'topk']),
    "gate_topk": hp.choice("gate_topk", [1, 2, 3, 4]),
}

# 目标函数（返回 -valid_f1 供 hyperopt 最小化）
def objective(param_dict):
    """返回 -valid_f1 供 hyperopt 最小化"""
    # ---- 固定随机种子确保可复现 ----
    random_seed = random.randint(1, 10000)
    print(f"Using random seed for trial: {random_seed}")
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # ---- 把 param_dict 写回 args ----
    for k, v in param_dict.items():
        setattr(args, k, v)

    # 创建模型
    model = MoEBaseModel(dataloader.kg_names, eval_ent, eval_rel, args)
    
    best_val_f1 = -1
    early_stop = 0

    for epoch in range(args.n_epoch):
        if early_stop > 3:  # 早停策略
            break
            
        # 训练一个epoch
        train_loss = model.train(train_pos, dataloader.KG_dict, ddi_graph)

        if (epoch + 1) % args.epoch_per_test == 0:
            # 验证集评估
            v_f1, v_acc, v_kap = model.evaluate(valid_pos, dataloader.vKG_dict, ddi_graph)
            
            # 记录调优过程中的指标
            model.log_metrics(epoch, v_f1, v_acc, v_kap, log_type="tuning_valid")
            
            # 更新学习率调度器
            model.update_scheduler(v_f1)
            
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                early_stop = 0
            else:
                early_stop += 1

    return {'loss': -best_val_f1, 'status': STATUS_OK}

# 运行超参数搜索
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=partial(tpe.suggest, n_startup_jobs=60),
            max_evals=100,
            trials=trials)

# 输出最佳超参数
print("\n============ Best hyper-parameters ============")
for k, v in best.items():
    print(f"{k}: {v}")

# 使用最佳超参数训练完整模型
print("\n============ Training with best parameters ============")
for k, v in best.items():
    setattr(args, k, v)

# 使用新的随机种子进行最终训练
final_seed = random.randint(1, 10000)
print(f"Using random seed for final training: {final_seed}")
random.seed(final_seed)
np.random.seed(final_seed)
torch.manual_seed(final_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(final_seed)
    
model = MoEBaseModel(dataloader.kg_names, eval_ent, eval_rel, args)
best_f1 = -1

# 记录最终使用的超参数到日志
moe_name = "_".join(dataloader.kg_names)
tuning_log = f"results/tuned_{moe_name}_hyperparams.txt"
with open(tuning_log, "w", encoding="utf-8") as f:
    f.write("最佳超参数:\n")
    for k, v in best.items():
        f.write(f"{k}: {v}\n")
    f.write(f"训练随机种子: {final_seed}\n")

for epoch in range(args.n_epoch):
    # 训练一个epoch
    train_loss = model.train(train_pos, dataloader.KG_dict, ddi_graph)
    
    if (epoch + 1) % args.epoch_per_test == 0:
        v_f1, v_acc, v_kap = model.evaluate(valid_pos, dataloader.vKG_dict, ddi_graph)
        t_f1, t_acc, t_kap = model.evaluate(test_pos, dataloader.tKG_dict, ddi_graph)
        print(f"[E{epoch}] valid-F1={v_f1:.4f} acc={v_acc:.4f} κ={v_kap:.4f} test-F1={t_f1:.4f} acc={t_acc:.4f} κ={t_kap:.4f}")
        
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
                print(f'>> Saved {ckpt}')
