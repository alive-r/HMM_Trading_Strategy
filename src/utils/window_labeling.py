import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def state_features(gamma: np.ndarray, r: pd.Series, tail=-0.02):
    """
    gamma: (T, K) 训练窗口职责；r: 与 gamma 对齐的日收益序列
    返回 shape (K, d) 的特征矩阵，每行对应一个状态的画像 f_k
    这里用 [mu, sigma, var5, rho] 作为例子
    """
    T, K = gamma.shape
    feats = []
    r1 = r.shift(-1)  # 下一期收益，用于决策/配资
    for k in range(K):
        w = gamma[:, k]
        w = w / (w.sum() + 1e-12)
        mu = np.nansum(w * r1.values)
        var = np.nansum(w * (r1.values - mu)**2)
        sigma = np.sqrt(max(var, 0.0))
        # 粗略 VaR（经验或高斯）
        var5 = np.nanpercentile(r1.values, 5)  # 简化；也可加权分位数
        # 简略趋势性（与上一期的相关）
        rho = np.corrcoef(r.values[:-1], r1.values[:-1])[0,1] if len(r)>2 else 0.0
        feats.append([mu, sigma, var5, rho])
    return np.array(feats)  # (K,4)

def build_cost(F_prev: np.ndarray, F_curr: np.ndarray, w=(1.0, 1.0, 0.5, 0.5), sticky=None):
    """
    代价矩阵：模板 vs 当前（越小越相似）
    w: 各特征权重；sticky: 可传入一个KxK矩阵，偏好保持上次映射（把保持不变的代价再减一点）
    """
    # 标准化各维度再加权欧氏
    mu = F_prev.mean(axis=0); sd = F_prev.std(axis=0) + 1e-9
    Fp = (F_prev - mu)/sd
    Fc = (F_curr - mu)/sd
    W = np.diag(w)
    # cost[i,j] = || W^{1/2} (Fp[i]-Fc[j]) ||
    diff = Fp[:,None,:] - Fc[None,:,:]  # (K,K,d)
    cost = np.sqrt(np.einsum('ijk,kl,ijk->ij', diff, W, diff))
    if sticky is not None:
        cost = cost - sticky  # 同一标签给个小优惠
    return cost

def window_labeling(gamma_train, r_train, prev_template=None, names_prev=None):
    """
    返回：names_curr（长度K的标签列表，如 ['MR','PA','TR']）、模板 F_curr（供下一窗口使用）
    """
    F_curr = state_features(gamma_train, r_train)  # (K,4)
    K = F_curr.shape[0]

    # 如果没有模板（第一窗口）：用单窗口规则先命名
    if prev_template is None:
        # 找 panic: sigma 最大 或 var5 最小
        sigma = F_curr[:,1]; var5 = F_curr[:,2]
        panic_idx = np.lexsort((var5, -sigma))[-1]  # 优先大sigma，其次小var5
        remain = [i for i in range(K) if i!=panic_idx]
        # 两者按 mu 排序：低→MR，高→TR
        mu = F_curr[:,0]
        mr_idx, tr_idx = sorted(remain, key=lambda i: mu[i])
        names = ['']*K
        names[mr_idx] = 'MR'; names[tr_idx] = 'TR'; names[panic_idx] = 'PA'
        return names, F_curr

    # 否则：做 Hungarian 匹配，让当前K个状态对齐到上一窗口的(MR,TR,PA)
    # 先把 prev_template 和 names_prev 排到固定顺序（MR,TR,PA）
    order = [names_prev.index(lbl) for lbl in ['MR','TR','PA'] if lbl in names_prev]
    F_prev = prev_template[order,:]
    names_target = [lbl for lbl in ['MR','TR','PA'] if lbl in names_prev]

    cost = build_cost(F_prev, F_curr, w=(1.0, 1.0, 0.7, 0.5))
    row_ind, col_ind = linear_sum_assignment(cost)
    names = ['']*K
    for r,c in zip(row_ind, col_ind):
        names[c] = names_target[r]  # 把目标标签赋给当前第 c 个状态

    return names, F_curr