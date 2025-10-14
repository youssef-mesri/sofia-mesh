#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent PPO Intelligent avec Masquage d'Actions
==============================================
Version compatible avec super_optimized_trainer.py
Utilise le masquage d'actions pour améliorer l'apprentissage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple

from ppo_dag_environment import DAGOptimizationEnv

class SmartPolicyNetwork(nn.Module):
    """Réseau de politique avec support du masquage d'actions"""
    
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        
        # Encodeur partagé
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Tête de politique (acteur)
        self.policy_head = nn.Linear(hidden_size, action_size)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(obs)
        action_logits = self.policy_head(encoded)
        return action_logits

class SmartValueNetwork(nn.Module):
    """Réseau de valeur séparé"""
    
    def __init__(self, obs_size: int, hidden_size: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)

class SmartDAGPPOAgent:
    """
    Agent PPO intelligent compatible avec super_optimized_trainer.py
    """
    def __init__(self, env: DAGOptimizationEnv, learning_rate: float, gamma: float, 
                 gae_lambda: float, clip_epsilon: float, entropy_coef: float, value_coef: float):
        
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparamètres
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Réseaux de neurones (noms compatibles avec le trainer)
        self.policy_network = SmartPolicyNetwork(env.obs_size, env.action_space.total_size).to(self.device)
        self.value_network = SmartValueNetwork(env.obs_size).to(self.device)
        
        # Optimiseurs (noms compatibles avec le trainer)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)

        # Buffers de mémoire
        self.reset_buffers()
        
        # Statistiques
        self.total_actions_taken = 0
        self.valid_actions_taken = 0
        
        print(f" Agent PPO intelligent initialisé (device: {self.device})")

    def reset_buffers(self):
        """Remet à zéro les buffers d'expérience"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        Sélectionne une action en utilisant le masquage.
        Retourne: (action, value, log_prob) - compatible avec super_optimized_trainer
        """
        self.total_actions_taken += 1
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Obtenir les logits de politique
            action_logits = self.policy_network(state_tensor)
            
            # Obtenir le masque d'actions valides
            action_mask = self.env.get_action_mask()
            action_mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
            
            # Appliquer le masque
            inf_tensor = torch.tensor(float('-inf'), device=self.device)
            masked_logits = torch.where(action_mask_tensor, action_logits, inf_tensor)
            
            # Créer la distribution
            action_dist = torch.distributions.Categorical(logits=masked_logits)
            
            # Valeur d'état
            state_value = self.value_network(state_tensor)
        
        if training:
            action = action_dist.sample()
        else:
            action = torch.argmax(action_dist.probs)
            
        log_prob = action_dist.log_prob(action)
        
        # Vérifier si l'action choisie est valide (pour les statistiques)
        if action_mask[action.item()]:
            self.valid_actions_taken += 1
            
        return action.item(), state_value.item(), log_prob.item()

    def store_transition(self, state, action, reward, value, log_prob, done):
        """Stocke une transition dans le buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Calcule les avantages avec GAE"""
        advantages = []
        returns = []
        
        gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value_t * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        return advantages, returns

    def update_policy(self, epochs: int = 10, batch_size: int = 64):
        """Met à jour la politique avec PPO"""
        
        if len(self.states) == 0:
            return
        
        # Calculer les avantages et retours
        advantages, returns = self.compute_gae()
        
        # Convertir en tenseurs
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normaliser les avantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Entraînement par epochs
        for epoch in range(epochs):
            # Mélanger les données
            indices = torch.randperm(len(states_tensor))
            
            for start_idx in range(0, len(states_tensor), batch_size):
                end_idx = min(start_idx + batch_size, len(states_tensor))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Forward pass
                action_logits = self.policy_network(batch_states)
                values = self.value_network(batch_states).squeeze()
                
                # Appliquer le masque pour chaque état dans le batch
                masked_logits = action_logits.clone()
                for i, state in enumerate(batch_states):
                    # Simuler le masque (simplification pour l'entraînement)
                    mask = torch.ones(action_logits.size(1), dtype=torch.bool, device=self.device)
                    inf_tensor = torch.tensor(float('-inf'), device=self.device)
                    masked_logits[i] = torch.where(mask, action_logits[i], inf_tensor)
                
                # Distribution et nouvelles log-probabilités
                action_dist = torch.distributions.Categorical(logits=masked_logits)
                new_log_probs = action_dist.log_prob(batch_actions)
                
                # Ratio de probabilité
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Loss PPO
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy = action_dist.entropy().mean()
                entropy_loss = -self.entropy_coef * entropy
                
                # Mise à jour du réseau de politique
                self.policy_optimizer.zero_grad()
                (policy_loss + entropy_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
                self.policy_optimizer.step()
                
                # Mise à jour du réseau de valeur
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
                self.value_optimizer.step()
        
        # Vider les buffers
        self.reset_buffers()

    def get_statistics(self) -> Dict:
        """Retourne les statistiques de l'agent - compatible avec super_optimized_trainer"""
        if self.total_actions_taken == 0:
            return {'valid_action_rate': 0.0, 'total_actions': 0}
        
        rate = self.valid_actions_taken / self.total_actions_taken
        stats = {
            'valid_action_rate': rate,
            'total_actions': self.total_actions_taken
        }
        
        # Réinitialiser pour la prochaine série d'épisodes
        self.total_actions_taken = 0
        self.valid_actions_taken = 0
        return stats
