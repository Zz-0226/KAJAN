import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from dataset import DubinsDataset
from pomo import InterleavedPOMONet
import time
import os
import math
import numpy as np

# =========================================================
# 🎯 核心贡献针对性消融实验 (Targeted Ablation Study)
# 围绕论文的三大核心设计展开，确保参数量与训练策略绝对公平
# =========================================================
ABLATION_MODE = 1

# 默认状态：Full Model (所有核心创新均开启)
cfg_gate = True
cfg_logit = True
cfg_lambda = 0.05

if ABLATION_MODE == 1:
    print("🚀 模式 1: KAJAN Full Model (完整版 - 包含门控、物理调制与曲率约束)")
elif ABLATION_MODE == 2:
    cfg_gate = False
    print("🚀 模式 2: w/o Gate (去除自适应门控，验证特征融合与搜索稳定性)")
elif ABLATION_MODE == 3:
    cfg_logit = False
    print("🚀 模式 3: w/o Physical Guidance (关闭物理引导概率调制，验证Dubins几何辅助的有效性)")
elif ABLATION_MODE == 4:
    cfg_lambda = 0.0
    print("🚀 模式 4: w/o Curvature Constraint (去除曲率约束，验证轨迹平滑对代价降低的贡献)")

save_name = f'history_mode_{ABLATION_MODE}.npy'
model_name = f'checkpoint_mode_{ABLATION_MODE}.pth'

CONFIG = {
    'num_cities': 30,
    'K': 16,
    'batch_size': 256,
    'epochs': 4000,

    # ⚡ 激进探索参数 (保留：解决前期长度下降太慢的问题)
    'lr': 3e-4,
    'entropy_weight': 0.05,
    'temperature_start': 2.0,
    'top_k_ratio': 0.20,

    # 动态消融开关
    'use_gating': cfg_gate,
    'use_logit_modulation': cfg_logit,
    'lambda_curv': cfg_lambda,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'history_save_path': save_name,
    'model_save_path': model_name,
    'dataset_uniform': 'train_balanced_30.pt',
    'dataset_cluster': 'train_clustered_30.pt'
}


def calc_dubins_reward(cities, angles, dubins_matrix, K):
    B, P, N = cities.shape
    device = cities.device
    flat_states = cities * K + angles
    flat_states_closed = torch.cat((flat_states, flat_states[:, :, 0:1]), dim=2)
    batch_idx_3d = torch.arange(B, device=device)[:, None, None].expand(B, P, N)
    u = flat_states_closed[:, :, :-1]
    v = flat_states_closed[:, :, 1:]
    return dubins_matrix[batch_idx_3d, u, v].sum(dim=2)


def train():
    device = CONFIG['device']
    ds_uniform = DubinsDataset(CONFIG['dataset_uniform'])
    ds_cluster = DubinsDataset(CONFIG['dataset_cluster'])
    mixed_dataset = ConcatDataset([ds_uniform, ds_cluster])

    dataloader = DataLoader(mixed_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    model = InterleavedPOMONet(embed_dim=128, K=CONFIG['K'], rho=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    cost_history = []
    start_epoch = 0
    best_length_so_far = 9999.0

    if os.path.exists(CONFIG['model_save_path']):
        print(f"🔄 恢复检查点: {CONFIG['model_save_path']}...")
        checkpoint = torch.load(CONFIG['model_save_path'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if os.path.exists(CONFIG['history_save_path']):
            cost_history = np.load(CONFIG['history_save_path']).tolist()
            if len(cost_history) > 0:
                best_length_so_far = min([h[0] for h in cost_history])
        print(f"▶️ 从 Epoch {start_epoch:04d} 继续训练。\n")

    for epoch in range(start_epoch, CONFIG['epochs']):
        model.train()
        epoch_loss, epoch_dubins_dist, epoch_total_cost = 0.0, 0.0, 0.0

        # =========================================================
        # 🚀 探索与退火策略 (所有模式严格一致)
        # =========================================================
        current_lambda = CONFIG['lambda_curv'] if epoch < 400 else max(0.0, CONFIG['lambda_curv'] * (
                    0.5 ** ((epoch - 400) / 100.0)))

        if epoch < 2500:
            new_lr = max(CONFIG['lr'] * (0.98 ** (epoch // 100)), 5e-5)
            current_entropy_weight = max(CONFIG['entropy_weight'] * (0.97 ** (epoch // 50)), 1e-3)
            current_temp = max(1.0, CONFIG['temperature_start'] * (0.95 ** (epoch // 100)))
        else:
            new_lr = max(5e-5 * (0.90 ** ((epoch - 2500) // 20)), 1e-6)
            current_entropy_weight = 0.0
            current_temp = max(0.1, 1.0 * (0.90 ** ((epoch - 2500) // 10)))

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        for coords, dubins_mat in dataloader:
            B, N, _ = coords.shape
            coords, dubins_mat = coords.to(device), dubins_mat.to(device)
            dubins_mat = torch.clamp(dubins_mat, max=99.0)

            optimizer.zero_grad()

            cities, angles, joint_log_probs, true_entropies = model(
                coords, dubins_mat, greedy=False, temperature=current_temp,
                use_gating=CONFIG['use_gating'],
                use_logit_modulation=CONFIG['use_logit_modulation']
            )

            P = cities.shape[1]
            dubins_cost = calc_dubins_reward(cities, angles, dubins_mat, CONFIG['K'])

            if current_lambda > 0:
                theta_t = angles.float() * (2 * math.pi / CONFIG['K'])
                theta_next = torch.roll(theta_t, shifts=-1, dims=2)
                coords_exp = coords.unsqueeze(1).expand(B, P, N, 2)
                curr_coords = coords_exp.gather(2, cities.unsqueeze(-1).expand(B, P, N, 2))
                next_cities = torch.roll(cities, shifts=-1, dims=2)
                next_coords = coords_exp.gather(2, next_cities.unsqueeze(-1).expand(B, P, N, 2))
                delta = next_coords - curr_coords
                phi_t = torch.atan2(delta[..., 1], delta[..., 0])
                C_turn = 1 - torch.cos(theta_next - theta_t)
                C_align = 1 - torch.cos(theta_t - phi_t)
                curvature_penalty = (0.7 * C_turn + 0.3 * C_align).sum(dim=2)
            else:
                curvature_penalty = 0.0

            total_cost = dubins_cost + current_lambda * curvature_penalty

            num_elites = max(1, int(total_cost.shape[1] * CONFIG['top_k_ratio']))
            elites, _ = torch.topk(total_cost, k=num_elites, dim=1, largest=False)
            baseline = elites.mean(dim=1, keepdim=True)

            advantage = baseline - total_cost
            adv_std = advantage.std(dim=1, keepdim=True)
            advantage = advantage / (adv_std + 1e-5)

            best_costs, best_idx = torch.min(total_cost, dim=1, keepdim=True)
            bonus = torch.zeros_like(advantage)
            bonus.scatter_(1, best_idx, 3.0)
            advantage = advantage + bonus

            entropy_loss = true_entropies.mean()
            policy_loss = -(advantage.detach() * joint_log_probs).mean()
            loss = policy_loss - current_entropy_weight * entropy_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 📊 同时记录纯 Dubins 距离和总优化目标
            epoch_loss += loss.item()
            epoch_dubins_dist += dubins_cost.min(dim=1)[0].mean().item()
            epoch_total_cost += total_cost.min(dim=1)[0].mean().item()

        avg_dubins = epoch_dubins_dist / len(dataloader)
        avg_total_cost = epoch_total_cost / len(dataloader)
        cost_history.append((avg_dubins, avg_total_cost))

        is_best = False
        if avg_dubins < best_length_so_far:
            best_length_so_far = avg_dubins
            is_best = True

        print(
            f"Epoch {epoch:04d} | Mode: {ABLATION_MODE} | L(Dubins): {avg_dubins:.4f} | Total Cost: {avg_total_cost:.4f} [Best L: {best_length_so_far:.4f}]")

        if (epoch + 1) % 10 == 0 or is_best:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_length': best_length_so_far
            }
            if (epoch + 1) % 10 == 0:
                torch.save(save_dict, CONFIG['model_save_path'])
                np.save(CONFIG['history_save_path'], np.array(cost_history))
            if is_best:
                torch.save(save_dict, CONFIG['model_save_path'].replace('.pth', '_best.pth'))


if __name__ == "__main__":
    train()