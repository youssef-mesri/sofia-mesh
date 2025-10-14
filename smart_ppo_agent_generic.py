import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -----------------------
# Policy network
# -----------------------
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, act_size, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        input_dim = obs_size
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, act_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# -----------------------
# Value network
# -----------------------
class ValueNetwork(nn.Module):
    def __init__(self, obs_size, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        input_dim = obs_size
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)


# -----------------------
# Smart PPO Agent
# -----------------------
class SmartPPOAgent:
    def __init__(self, env, hidden_sizes=(128, 128), device="cpu"):
        """
        Agent PPO générique avec masquage d'actions.
        Compatible avec DAGOptimizationEnv et MeshPatchEnvCustom.
        """
        self.env = env
        self.obs_size = env.obs_size
        self.act_size = env.action_space.total_size
        self.device = device

        self.policy = PolicyNetwork(self.obs_size, self.act_size, hidden_sizes).to(device)
        self.value_function = ValueNetwork(self.obs_size, hidden_sizes).to(device)

    def get_action_masked_logits(self, obs, mask):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(obs_t).squeeze(0)

        # Masquage d'actions invalides
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        masked_logits = logits.clone()
        masked_logits[~mask_t] = -1e9  # probabilité quasi-nulle
        return masked_logits

    def select_action(self, obs, mask):
        logits = self.get_action_masked_logits(obs, mask)
        probs = torch.distributions.Categorical(logits=logits)
        action = probs.sample()
        logp = probs.log_prob(action)
        return int(action.item()), float(logp.item())

    def get_value(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.value_function(obs_t).item()

    def compute_policy_loss(self, obs_batch, act_batch, adv_batch, logp_old_batch):
        logits = self.policy(obs_batch)
        probs = torch.distributions.Categorical(logits=logits)
        logp = probs.log_prob(act_batch)

        ratio = torch.exp(logp - logp_old_batch)
        clip_adv = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv_batch
        loss_pi = -(torch.min(ratio * adv_batch, clip_adv)).mean()

        approx_kl = (logp_old_batch - logp).mean().item()
        return loss_pi, approx_kl

    def compute_value_loss(self, obs_batch, ret_batch):
        value = self.value_function(obs_batch)
        return F.mse_loss(value, ret_batch)
