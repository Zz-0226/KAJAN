import os

# ⚡ 终极死锁修复
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import numpy as np
import math
import multiprocessing
import uuid
from torch.utils.data import Dataset
from tqdm import tqdm

CONFIG = {
    'K': 16,
    'rho': 0.1,
    'suites': [
        # --- 核心训练集 (N=30) ---
        {'N': 30, 'num': 5000, 'dist': 'uniform', 'file': 'train_balanced_30.pt'},
        {'N': 30, 'num': 5000, 'dist': 'clustered', 'file': 'train_clustered_30.pt'},

        # 🚀 全面升级：千级别黄金标准测试集 🚀
        # --- 向下泛化测试集 (N=10) ---
        {'N': 10, 'num': 1000, 'dist': 'uniform', 'file': 'test_uniform_10.pt'},
        {'N': 10, 'num': 1000, 'dist': 'clustered', 'file': 'test_clustered_10.pt'},

        # --- 基准测试集 (N=30) ---
        {'N': 30, 'num': 1000, 'dist': 'uniform', 'file': 'test_uniform_30.pt'},
        {'N': 30, 'num': 1000, 'dist': 'clustered', 'file': 'test_clustered_30.pt'},

        # --- 向上泛化测试集 (N=50) ---
        {'N': 50, 'num': 1000, 'dist': 'uniform', 'file': 'test_uniform_50.pt'},
        {'N': 50, 'num': 1000, 'dist': 'clustered', 'file': 'test_clustered_50.pt'}
    ]
}


class DubinsDataset(Dataset):
    def __init__(self, filepath):
        super().__init__()
        # ⚡ 修复：移除了 weights_only=False，兼容旧版 PyTorch
        data = torch.load(filepath)

        # 1. 提取坐标
        if 'coords' in data:
            self.coords = data['coords']
        elif 'nodes' in data:
            self.coords = data['nodes']
        else:
            raise KeyError(f"❌ 找不到坐标数据！文件包含的键有: {list(data.keys() if isinstance(data, dict) else type(data))}")

        # 2. 提取矩阵
        if 'dubins_matrices' in data:
            self.dubins_matrices = data['dubins_matrices']
        elif 'dubins_mat' in data:
            self.dubins_matrices = data['dubins_mat']
        elif 'dubins_matrix' in data:
            self.dubins_matrices = data['dubins_matrix']
        elif 'distance_matrix' in data:
            self.dubins_matrices = data['distance_matrix']
        elif 'dist_mat' in data:
            self.dubins_matrices = data['dist_mat']
        else:
            raise KeyError(f"❌ 找不到距离矩阵！当前文件包含的键有: {list(data.keys() if isinstance(data, dict) else type(data))}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.dubins_matrices[idx]

def mod2pi(theta):
    return theta - 2.0 * math.pi * math.floor(theta / (2.0 * math.pi))

def dubins_LSL(alpha, beta, d):
    p_sq = 2 + d * d - (2 * math.cos(alpha - beta)) + (2 * d * (math.sin(alpha) - math.sin(beta)))
    if p_sq < 0: return None
    tmp = math.atan2((math.cos(beta) - math.cos(alpha)), d + math.sin(alpha) - math.sin(beta))
    return mod2pi(-alpha + tmp) + math.sqrt(p_sq) + mod2pi(beta - tmp)

def dubins_RSR(alpha, beta, d):
    p_sq = 2 + d * d - (2 * math.cos(alpha - beta)) + (2 * d * (math.sin(beta) - math.sin(alpha)))
    if p_sq < 0: return None
    tmp = math.atan2((math.cos(alpha) - math.cos(beta)), d - math.sin(alpha) + math.sin(beta))
    return mod2pi(alpha - tmp) + math.sqrt(p_sq) + mod2pi(-beta + tmp)

def dubins_LSR(alpha, beta, d):
    p_sq = -2 + d * d + (2 * math.cos(alpha - beta)) + (2 * d * (math.sin(alpha) + math.sin(beta)))
    if p_sq < 0: return None
    p = math.sqrt(p_sq)
    tmp = math.atan2((-math.cos(alpha) - math.cos(beta)), d + math.sin(alpha) + math.sin(beta)) - math.atan2(-2.0, p)
    return mod2pi(-alpha + tmp) + p + mod2pi(-beta + tmp)

def dubins_RSL(alpha, beta, d):
    p_sq = -2 + d * d + (2 * math.cos(alpha - beta)) - (2 * d * (math.sin(alpha) + math.sin(beta)))
    if p_sq < 0: return None
    p = math.sqrt(p_sq)
    tmp = math.atan2((math.cos(alpha) + math.cos(beta)), d - math.sin(alpha) - math.sin(beta)) - math.atan2(2.0, p)
    return mod2pi(alpha - tmp) + p + mod2pi(beta - tmp)

def dubins_RLR(alpha, beta, d):
    tmp = (6.0 - d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))) / 8.0
    if abs(tmp) > 1.0: return None
    p = mod2pi(2.0 * math.pi - math.acos(tmp))
    t = mod2pi(alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta)) + p / 2.0)
    return t + p + mod2pi(alpha - beta - t + p)

def dubins_LRL(alpha, beta, d):
    tmp = (6.0 - d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (-math.sin(alpha) + math.sin(beta))) / 8.0
    if abs(tmp) > 1.0: return None
    p = mod2pi(2.0 * math.pi - math.acos(tmp))
    t = mod2pi(-alpha - math.atan2(math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)) + p / 2.0)
    return t + p + mod2pi(beta - alpha - t + p)

def get_dubins_length(q1, q2, rho):
    dx, dy = q2[0] - q1[0], q2[1] - q1[1]
    D = math.sqrt(dx * dx + dy * dy)
    d = D / rho
    theta = mod2pi(math.atan2(dy, dx))
    alpha = mod2pi(q1[2] - theta)
    beta = mod2pi(q2[2] - theta)

    best_len = float('inf')
    for method in (dubins_LSL, dubins_RSR, dubins_LSR, dubins_RSL, dubins_RLR, dubins_LRL):
        L = method(alpha, beta, d)
        if L is not None and L < best_len: best_len = L
    return best_len * rho if best_len != float('inf') else float('inf')

def generate_single_instance(args):
    N, K, rho, seed_int, dist_type = args
    rng = np.random.default_rng(seed_int)

    if dist_type == 'uniform':
        coords = rng.random((N, 2))
    elif dist_type == 'clustered':
        centers = rng.random((3, 2))
        cluster_assignments = rng.integers(0, 3, size=N)
        noise = rng.normal(loc=0.0, scale=0.1, size=(N, 2))
        coords = np.clip(centers[cluster_assignments] + noise, 0.0, 1.0)
    else:
        coords = rng.random((N, 2))

    angles = np.linspace(0, 2 * math.pi, K, endpoint=False)
    dist_mat = np.full((N * K, N * K), np.inf)
    all_poses = [(coords[c][0], coords[c][1], angles[a]) for c in range(N) for a in range(K)]

    for u, q1 in enumerate(all_poses):
        c1 = u // K
        for v, q2 in enumerate(all_poses):
            c2 = v // K
            if c1 == c2: continue
            dist_mat[u, v] = get_dubins_length(q1, q2, rho)

    return coords, dist_mat

def build_dataset(num_instances, filename, N, K, rho, dist_type):
    if os.path.exists(filename):
        print(f"⏭️ {filename} 已存在，跳过生成。")
        return

    print(f"\n🚀 开始生成 [{dist_type}] 分布集 (N={N}, Instances={num_instances})...")
    seeds = [int(uuid.uuid4().int & (1 << 32) - 1) for _ in range(num_instances)]
    args_list = [(N, K, rho, seeds[i], dist_type) for i in range(num_instances)]

    coords_list, matrix_list = [], []
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    chunk_size = max(1, num_instances // (num_cores * 10))

    with multiprocessing.Pool(num_cores) as pool:
        for coords, dist_mat in tqdm(pool.imap_unordered(generate_single_instance, args_list, chunksize=chunk_size),
                                     total=num_instances):
            coords_list.append(coords)
            matrix_list.append(dist_mat)

    torch.save({
        'coords': torch.tensor(np.array(coords_list), dtype=torch.float32),
        'dubins_matrix': torch.tensor(np.array(matrix_list), dtype=torch.float32)
    }, filename)
    print(f"✅ 生成完毕！保存至: {filename}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    for suite in CONFIG['suites']:
        build_dataset(suite['num'], suite['file'], suite['N'], CONFIG['K'], CONFIG['rho'], suite['dist'])