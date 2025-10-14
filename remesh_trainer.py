import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from remesh_environment import MeshPatchEnvCustom, mesh_min_angle
from smart_ppo_agent_generic import SmartPPOAgent
import torch
import torch.optim as optim
from tqdm import trange


# -----------------------
# Config
# -----------------------
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_epochs = 50
        self.steps_per_epoch = 200
        self.batch_size = 64
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.pi_lr = 3e-4
        self.vf_lr = 1e-3
        self.train_pi_iters = 50
        self.train_v_iters = 50
        self.lam = 0.97
        self.target_kl = 0.01
        self.max_steps = 100
        self.vis_interval = 10   # fréquence de sauvegarde images
        self.plot_rewards = True


# -----------------------
# Trainer
# -----------------------
class SuperOptimizedPPOTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.env = None
        self.agent = None
        self.optimizer_pi = None
        self.optimizer_vf = None
        self.reward_history = []
        self.quality_history = []

    def build_initial_mesh(self, npts=50, seed=0):
        rng = np.random.RandomState(seed)
        pts = rng.rand(npts, 2)
        corners = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        pts = np.vstack((pts, corners))
        tri = Delaunay(pts)
        return pts, tri.simplices.copy()

    def setup_components(self):
        # --- build env ---
        pts, tris = self.build_initial_mesh(npts=50, seed=2025)
        self.env = MeshPatchEnvCustom(
            points=pts,
            triangles=tris,
            max_steps=self.config.max_steps,
            max_points=200,
        )

        # --- build agent ---
        self.agent = SmartPPOAgent(
            env=self.env,
            hidden_sizes=(128, 128),
            device=self.device,
        )

        self.optimizer_pi = optim.Adam(self.agent.policy.parameters(), lr=self.config.pi_lr)
        self.optimizer_vf = optim.Adam(self.agent.value_function.parameters(), lr=self.config.vf_lr)

        # save initial mesh
        self.save_mesh_plot(epoch=0, title="Initial mesh")

    def save_mesh_plot(self, epoch, title="Mesh"):
        plt.figure(figsize=(5, 5))
        plt.triplot(self.env.points[:, 0], self.env.points[:, 1], self.env.triangles, linewidth=0.6)
        plt.scatter(self.env.points[:, 0], self.env.points[:, 1], s=6)
        plt.gca().set_aspect("equal")
        plt.title(f"{title} (epoch {epoch})")
        fname = f"mesh_epoch_{epoch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"[Viz] Saved mesh to {fname}")

    def save_progress_plot(self):
        if not self.reward_history:
            return
        plt.figure(figsize=(6, 4))
        plt.plot(self.reward_history, marker="o", label="Avg reward")
        plt.plot(self.quality_history, marker="s", label="Min angle (quality)")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training progression")
        plt.legend()
        plt.grid(True)
        plt.savefig("progress_vs_epoch.png", dpi=150)
        plt.close()
        print("[Viz] Saved progress curve to progress_vs_epoch.png")

    def train(self):
        self.setup_components()

        for epoch in trange(self.config.total_epochs, desc="Training epochs"):
            obs = self.env.reset()
            ep_rewards = []
            batch_obs, batch_acts, batch_weights, batch_rtgs, batch_vals, batch_logp = [], [], [], [], [], []
            ep_ret, ep_len = 0, 0

            for step in range(self.config.steps_per_epoch):
                mask = self.env.get_action_mask()
                act, logp = self.agent.select_action(obs, mask)

                next_obs, reward, done, info = self.env.step(act)
                val = self.agent.get_value(obs)

                batch_obs.append(obs)
                batch_acts.append(act)
                batch_weights.append(reward)
                batch_rtgs.append(reward)  # simple reward-to-go placeholder
                batch_vals.append(val)
                batch_logp.append(logp)

                obs = next_obs
                ep_ret += reward
                ep_len += 1

                if done or (step == self.config.steps_per_epoch - 1):
                    obs = self.env.reset()
                    ep_rewards.append(ep_ret)
                    ep_ret, ep_len = 0, 0

            # Convert to tensors
            batch_obs = torch.as_tensor(np.array(batch_obs), dtype=torch.float32, device=self.device)
            batch_acts = torch.as_tensor(np.array(batch_acts), dtype=torch.int64, device=self.device)
            batch_weights = torch.as_tensor(np.array(batch_weights), dtype=torch.float32, device=self.device)
            batch_vals = torch.as_tensor(np.array(batch_vals), dtype=torch.float32, device=self.device)
            batch_logp = torch.as_tensor(np.array(batch_logp), dtype=torch.float32, device=self.device)

            # Compute advantages
            adv = batch_weights - batch_vals.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Update policy
            for _ in range(self.config.train_pi_iters):
                pi_loss, approx_kl = self.agent.compute_policy_loss(
                    batch_obs, batch_acts, adv, batch_logp
                )
                self.optimizer_pi.zero_grad()
                pi_loss.backward()
                self.optimizer_pi.step()
                if approx_kl > 1.5 * self.config.target_kl:
                    break

            # Update value function
            for _ in range(self.config.train_v_iters):
                v_loss = self.agent.compute_value_loss(batch_obs, batch_rtgs)
                self.optimizer_vf.zero_grad()
                v_loss.backward()
                self.optimizer_vf.step()

            avg_reward = np.mean(ep_rewards) if ep_rewards else 0
            self.reward_history.append(avg_reward)

            # mesurer la qualité du maillage (min angle global)
            mesh_quality = mesh_min_angle(self.env.points, self.env.triangles)
            self.quality_history.append(mesh_quality)

            print(f"Epoch {epoch}: avg_reward={avg_reward:.3f}, min_angle={mesh_quality:.2f}°")

            # save visualization every N epochs
            if epoch % self.config.vis_interval == 0 or epoch == self.config.total_epochs - 1:
                self.save_mesh_plot(epoch, title="Mesh state")

        # Save progression curve
        if self.config.plot_rewards:
            self.save_progress_plot()


if __name__ == "__main__":
    config = Config()
    trainer = SuperOptimizedPPOTrainer(config)
    trainer.train()
