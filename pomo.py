import torch
import torch.nn as nn
import math


# =====================================================================
# 1. 编码器层 (保留参数公平的门控消融)
# =====================================================================
class GatedEdgeAwareEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_hidden=512):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.edge_proj = nn.Linear(embed_dim, num_heads)

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[0].bias, 2.0)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        )

    def forward(self, x, edge_emb, use_gating=True):
        B, N, C = x.shape
        head_dim = C // self.num_heads

        q = self.q_proj(x).view(B, N, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # 默认使用边缘特征偏置
        edge_bias = self.edge_proj(edge_emb).permute(0, 3, 1, 2)
        attn_scores = attn_scores + edge_bias

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v).transpose(1, 2).reshape(B, N, C)

        # ⚡ Gating 消融：无论开关与否，前向计算均执行，保证网络结构/显存对齐
        gate_input = torch.cat([x, attn_out], dim=-1)
        gate_val = self.gate(gate_input)

        if not use_gating:
            # 去除门控作用：用全 1 张量代替，退化为普通残差连接
            gate_val = torch.ones_like(gate_val)

        x = self.norm1(x + gate_val * self.out_proj(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


# =====================================================================
# 2. 解码器层 (物理引导约束与概率调制消融)
# =====================================================================
class TrueJointActionDecoder(nn.Module):
    def __init__(self, embed_dim, K, rho=0.1):
        super().__init__()
        self.K = K
        self.C = 10.0
        self.rho = rho

        self.angle_embedder = nn.Sequential(
            nn.Linear(2, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, embed_dim)
        )
        angles_rad = torch.arange(K, dtype=torch.float32) * (2 * math.pi / K)
        self.register_buffer('base_angle_features', torch.stack([torch.sin(angles_rad), torch.cos(angles_rad)], dim=-1))

        self.project_context = nn.Linear(embed_dim * 3, embed_dim)
        self.joint_feature_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.W_query = nn.Linear(embed_dim, embed_dim)
        self.K_joint = nn.Linear(embed_dim, embed_dim)

        self.oracle_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def _get_angle_embedding(self, angle_indices):
        B, P = angle_indices.shape
        features = self.base_angle_features.unsqueeze(0).unsqueeze(0).expand(B, P, self.K, 2)
        idx_exp = angle_indices.unsqueeze(-1).unsqueeze(-1).expand(B, P, 1, 2)
        return self.angle_embedder(features.gather(2, idx_exp).squeeze(2))

    def forward(self, global_embed, node_embeddings, dubins_matrix,
                greedy=False, temperature=1.0, use_logit_modulation=True):

        B, N, C = node_embeddings.shape
        P = N
        device = node_embeddings.device

        angle_embs = self.angle_embedder(self.base_angle_features)
        node_exp_for_joint = node_embeddings.unsqueeze(2).expand(B, N, self.K, C)
        angle_exp_for_joint = angle_embs.unsqueeze(0).unsqueeze(0).expand(B, N, self.K, C)

        # 默认开启联合特征组合
        raw_joint_features = torch.cat([node_exp_for_joint, angle_exp_for_joint], dim=-1)
        joint_keys = self.K_joint(self.joint_feature_mlp(raw_joint_features))

        first_city = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        current_angle = torch.zeros((B, P), dtype=torch.long, device=device) if greedy else torch.randint(0, self.K,
                                                                                                          (B, P),
                                                                                                          device=device)

        selected_cities, selected_angles = [first_city], [current_angle]
        joint_log_probs, true_entropies = [], []

        city_mask = torch.zeros(B, P, N, dtype=torch.bool, device=device).scatter(2, first_city.unsqueeze(-1), True)

        node_emb_exp = node_embeddings.unsqueeze(1).expand(B, P, N, C)
        global_emb_exp = global_embed.unsqueeze(1).expand(B, P, C)

        current_node_emb = node_emb_exp.gather(2, first_city.unsqueeze(-1).unsqueeze(-1).expand(B, P, 1, C)).squeeze(2)
        current_angle_emb = self._get_angle_embedding(current_angle)

        batch_idx = torch.arange(B, device=device)[:, None].expand(B, P)

        for step in range(1, N):
            context = torch.cat([global_emb_exp, current_node_emb, current_angle_emb], dim=-1)
            query = self.W_query(self.project_context(context))

            joint_logits = torch.einsum('bpc,bnkc->bpnk', query, joint_keys) / math.sqrt(C)
            joint_logits = self.C * torch.tanh(joint_logits / self.C)

            # ⚡ Physics Modulation 消融
            if use_logit_modulation:
                curr_state_flat = selected_cities[-1] * self.K + selected_angles[-1]
                exact_costs = dubins_matrix[batch_idx, curr_state_flat, :].view(B, P, N, self.K)
                oracle_penalty = torch.relu(self.oracle_mlp(exact_costs.unsqueeze(-1)).squeeze(-1))
                joint_logits = joint_logits - oracle_penalty

            joint_logits_flat = joint_logits.view(B, P, N * self.K)
            expanded_mask = city_mask.unsqueeze(-1).expand(B, P, N, self.K).reshape(B, P, N * self.K)
            joint_logits_flat = joint_logits_flat.masked_fill(expanded_mask, float('-inf'))

            joint_logits_flat = joint_logits_flat / temperature
            joint_probs_flat = torch.softmax(joint_logits_flat, dim=-1)

            if greedy:
                action_idx = joint_probs_flat.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(joint_probs_flat)
                action_idx = dist.sample()

            city = action_idx // self.K
            angle = action_idx % self.K

            step_log_prob = torch.log(joint_probs_flat + 1e-10).gather(-1, action_idx.unsqueeze(-1)).squeeze(-1)
            step_entropy = -(joint_probs_flat * torch.log(joint_probs_flat + 1e-10)).sum(dim=-1)

            selected_cities.append(city)
            selected_angles.append(angle)
            joint_log_probs.append(step_log_prob)
            true_entropies.append(step_entropy)

            city_mask = city_mask.scatter(2, city.unsqueeze(-1), True)
            current_node_emb = node_emb_exp.gather(2, city.unsqueeze(-1).unsqueeze(-1).expand(B, P, 1, C)).squeeze(2)
            current_angle_emb = self._get_angle_embedding(angle)

        zero_tensor = torch.zeros((B, P), device=device)
        joint_log_probs = [zero_tensor] + joint_log_probs
        true_entropies = [zero_tensor] + true_entropies

        return torch.stack(selected_cities, dim=2), \
            torch.stack(selected_angles, dim=2), \
            torch.stack(joint_log_probs, dim=2).sum(dim=2), \
            torch.stack(true_entropies, dim=2).sum(dim=2)


# =====================================================================
# 3. 顶层网络
# =====================================================================
class InterleavedPOMONet(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=6, K=16, rho=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = K
        self.init_embed = nn.Linear(2, embed_dim)
        self.edge_compressor = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, embed_dim)
        )
        self.encoder_layers = nn.ModuleList([
            GatedEdgeAwareEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.decoder = TrueJointActionDecoder(embed_dim, K, rho)

    def forward(self, coords, dubins_matrix, greedy=False, temperature=1.0,
                use_gating=True, use_logit_modulation=True):
        B, N, _ = coords.shape
        node_embeddings = self.init_embed(coords)

        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = diff.norm(p=2, dim=-1, keepdim=True)
        direction = diff / (dist + 1e-8)
        raw_edge_features = torch.cat([dist, direction], dim=-1)
        edge_embeddings = self.edge_compressor(raw_edge_features)

        for layer in self.encoder_layers:
            node_embeddings = layer(node_embeddings, edge_embeddings, use_gating=use_gating)

        global_embed = node_embeddings.mean(dim=1)

        return self.decoder(
            global_embed,
            node_embeddings,
            dubins_matrix,
            greedy=greedy,
            temperature=temperature,
            use_logit_modulation=use_logit_modulation
        )