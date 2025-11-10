import torch
from pykeops.numpy import LazyTensor
import numpy as np
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

import torch
import torch.distributed as dist

def all_gather_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    # 分布式环境下执行all_gather
    world_size = dist.get_world_size()
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor.contiguous())  # contiguous确保内存连续
    return torch.cat(tensor_list, dim=0)


class DTAMetrics(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)


    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # 确保输入为一维张量 (batch_size,)
        self.preds.append(preds.detach().flatten())
        self.targets.append(targets.detach().flatten())

    def compute(self) -> dict:
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)
        preds = all_gather_if_needed(preds)
        targets = all_gather_if_needed(targets)
        results = {
            "CI": self._calculate_ci_keops2(preds, targets),
            "MSE": self._calculate_mse(preds, targets),
            "RMSE": self._calculate_rmse(preds, targets),
            "RM2": self._calculate_rm2(preds, targets),
            "Pearson": self._calculate_pearson(preds, targets),
            "Spearman": self._calculate_spearman(preds, targets)
        }
        return results

    def _calculate_ci(self, preds, targets) -> torch.Tensor:
        """计算最终的CI分数，使用向量操作优化效率"""
        # 合并所有批次数据 (总样本数,)
        device = preds.device
        n = len(targets)

        if n < 2:
            return torch.tensor(0.5, device=device)  # 样本不足时返回随机水平

        concordant = torch.tensor(0.0, device=device)
        tied = torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)

        # 单循环 + 向量操作（替代双层循环）
        for i in range(n):
            gt_i = targets[i]
            pred_i = preds[i]
            # 后续所有样本（j > i）的真实值和预测值（向量）
            gt_j = targets[i + 1:]
            pred_j = preds[i + 1:]
            # 计算掩码：j的真实值 > i 或 < i
            mask_greater = gt_j > gt_i
            mask_less = gt_j < gt_i
            total_pairs = mask_greater.sum() + mask_less.sum()
            if total_pairs == 0:
                continue  # 无有效对比对，跳过
            # 计算一致对：(j真实值>i 且 j预测值>i) 或 (j真实值<i 且 j预测值<i)
            concordant += (mask_greater & (pred_j > pred_i)).sum()
            concordant += (mask_less & (pred_j < pred_i)).sum()
            # 计算打结对（0.5权重）
            tied += (mask_greater & (pred_j == pred_i)).sum() * 0.5
            tied += (mask_less & (pred_j == pred_i)).sum() * 0.5
            total += total_pairs
        # 避免除零错误
        if total == 0:
            return torch.tensor(0.5, device=device)
        # 计算最终CI
        return (concordant + tied) / total

    def _calculate_ci_keops(self, preds, targets):
        from pykeops.torch import Vi, Vj
        preds = preds[:, None]
        targets = targets[:, None]
        g = Vi(preds) - Vj(preds)
        g = (g == 0) * 0.5 + (g > 0) * 1.0
        f = Vi(targets) - Vj(targets)
        f = f > 0
        numerator = (g * f).sum(0).sum()
        denominator = f.sum(0).sum()
        if denominator==0:
            return torch.tensor(0.5, device=preds.device)
        return numerator / denominator

    def _calculate_ci_keops2(self, preds, targets):
        from pykeops.torch import Vi, Vj
        preds = preds[:, None]
        targets = targets[:, None]
        gt_mask = Vi(targets) > Vj(targets)
        diff = Vi(preds) - Vj(preds)
        h_one = (diff > 0)
        h_half = (diff == 0)
        numerator = (gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5).sum(0).sum()
        denominator = gt_mask.sum(0).sum()
        if denominator == 0:
            return torch.tensor(0.5, device=preds.device)
        return numerator / denominator


    def _calculate_mse(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean((preds - targets) **2)

    def _calculate_rmse(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算RMSE"""
        mse = self._calculate_mse(preds, targets)
        return torch.sqrt(mse)

    def _calculate_pearson(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """使用torch.corrcoef计算Pearson相关系数（更简洁高效）"""
        if len(preds) < 2:
            return torch.tensor(0.0, device=preds.device)
        combined = torch.stack([preds, targets], dim=0)
        corr_matrix = torch.corrcoef(combined)
        pearson = corr_matrix[0, 1]
        return pearson if not torch.isnan(pearson) else torch.tensor(0.0, device=preds.device)

    def _calculate_spearman(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        def rank(x):
            _, indices = torch.sort(x)
            ranks = torch.zeros_like(indices, device=x.device)
            ranks[indices] = torch.arange(len(x), device=x.device)
            return ranks
        rank_preds = rank(preds)
        rank_targets = rank(targets)
        return self._calculate_pearson(rank_preds, rank_targets)  # 直接复用Pearson方法

    def _calculate_rm2(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        r2 = self._calculate_pearson(preds, targets)**2
        k = (preds*targets).sum() /(preds*preds).sum()
        upp = ((targets-k*preds)**2).sum()
        down= ((targets - targets.mean())**2).sum()
        r02 = 1-upp/down
        rm2 = r2 * (1 - torch.sqrt(torch.absolute((r2*r2)-(r02*r02))))
        return rm2


def cindex_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    if y_true.dim() > 1:
        y_true = y_true.squeeze(-1)
    if y_pred.dim() > 1:
        y_pred = y_pred.squeeze(-1)
    g = y_pred.unsqueeze(-1) - y_pred
    g = (torch.eq(g, 0.0).float() * 0.5) + (torch.gt(g, 0.0).float() * 1.0)
    f = y_true.unsqueeze(-1) - y_true
    f = torch.gt(f, 0.0)
    # f = torch.tril(f.float(), diagonal=-1)
    numerator = torch.sum(g * f)
    denominator = torch.sum(f)
    cindex = torch.where(torch.eq(denominator, 0), torch.tensor(0.0, device=y_true.device), numerator / denominator)
    return cindex

def get_cindex(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = (gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5).sum() / (gt_mask).sum()
    return CI

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))


def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

if __name__ == "__main__":
    preds = torch.randn(50000)
    targets = torch.randn_like(preds)
    from time import time

    metrics = DTAMetrics()
    start_time = time()

    print(metrics._calculate_rm2(preds, targets), time()-start_time)
    start_time = time()
    print(get_rm2(preds.numpy(), targets.numpy()), time()-start_time)
    exit()
    metrics.update(preds, targets)
    print(metrics.compute(), time() - start_time)
    start_time = time()
    print(metrics._calculate_ci_keops(preds, targets), time() - start_time)
    start_time = time()
    print(metrics._calculate_ci_keops2(preds, targets), time() - start_time)
    start_time = time()
    print(metrics._calculate_ci(preds, targets), time() - start_time)
    start_time = time()
    print(cindex_score(targets, preds), time() - start_time)
    start_time = time()
    print(get_cindex(targets, preds), time() - start_time)

