# train_rl_mpc.py
import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from mpc_v1 import RobustMPC, TrajectoryType, generate_reference, A_delta, B_delta, D_delta
import numpy as np
import torch

# 自定义 RL 环境
class MPCRLEnv(gym.Env):
    def __init__(self):
        super(MPCRLEnv, self).__init__()
        self.mpc = RobustMPC()
        self.traj_type = TrajectoryType.DOUBLE_LANE
        self.sim_time = 30.0
        self.Ts = 0.05
        self.t = np.arange(0, self.sim_time, self.Ts)
        self.x_ref, self.y_ref, self.dx_ref, self.dy_ref, self.ddx_ref, self.ddy_ref = generate_reference(self.traj_type, self.t)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,))
        self.action_space = gym.spaces.Box(low=0.5, high=2.0, shape=(3,))  # [Q_scale, R_scale, S_scale]
        self.reset()

    def reset(self):
        self.xi = np.zeros(5)  # 状态: [e_x, Δe_x, e_y, Δe_y, I_e]
        self.k = 0
        self.prev_error = np.zeros(2)
        return self._normalize_state(self.xi)

    def step(self, action):
        # 反归一化动作
        action = np.clip(action, 0.5, 2.0)
        
        # 更新 MPC 参数
        Q_scale, R_scale, S_scale = action
        self.mpc.Q = np.diag([50*Q_scale, 5*Q_scale, 50*Q_scale, 5*Q_scale, 0.1*Q_scale])
        self.mpc.R = np.diag([0.01*R_scale, 0.01*R_scale])
        self.mpc.S = np.diag([0.1*S_scale, 0.1*S_scale])

        # 计算误差导数
        current_error = np.array([self.xi[0], self.xi[2]])
        de = (current_error - self.prev_error) / self.Ts if self.k > 0 else np.zeros(2)
        self.prev_error = current_error

        # 调用滑模观测器
        d_hat = self.mpc.obs.update(current_error, de)

        # 求解 MPC
        xi_ref = np.zeros((15, 5))  # 预测时域 N=15
        u_opt = self.mpc.solve(self.xi, xi_ref, d_hat)

        # 状态更新
        u_ff = np.array([self.ddx_ref[self.k], self.ddy_ref[self.k]])
        u_total = u_opt + 0.5 * u_ff
        xi_next = A_delta @ self.xi + B_delta @ u_total + D_delta @ np.array([self.ddx_ref[self.k], self.ddy_ref[self.k]])

        # 更新状态到下一时刻
        self.xi = xi_next.copy() 

        # 记录奖励 (负的跟踪误差)
        error_penalty = - (self.xi[0]**2 + self.xi[2]**2)
        control_penalty = - 0.001 * np.sum(u_total**2)  # 控制输入平滑性
        action_penalty = - 0.01 * np.sum((action - 1.0)**2)  # 鼓励参数接近初始值
        reward = error_penalty + control_penalty + action_penalty

        # 判断是否终止
        self.k += 1
        done = self.k >= len(self.t)
        
        # 归一化下一状态
        next_state = self._normalize_state(xi_next)
        
        # 增强状态信息
        augmented_state = np.concatenate([
            self.xi,
            d_hat,  # 扰动估计
            [self.k / len(self.t)]  # 时间进度
        ])
        return self._normalize_state(augmented_state), reward, done, {}
    
    def _normalize_state(self, state):
        # 假设误差范围 [-1.0, 1.0]，积分项范围 [-I_max, I_max]
        normalized = np.array([
            state[0] / 1.0,   # e_x
            state[1] / 10.0,  # Δe_x
            state[2] / 1.0,   # e_y
            state[3] / 10.0,  # Δe_y
            state[4] / 1.0  # I_e
        ])
        return np.clip(normalized, -1.0, 1.0)

class TrainingMonitorCallback(BaseCallback):
    def __init__(self, check_freq=100, verbose=0):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.q_values = []
        # 初始化实时绘图
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 12))
        
    def _on_step(self) -> bool:
        # 每 check_freq 步记录一次数据
        if self.n_calls % self.check_freq == 0:
            # 记录当前奖励（从环境获取）
            if "info" in self.locals and "episode" in self.locals["info"]:
                self.rewards.append(self.locals["info"]["episode"]["r"])
            
            # 记录 Q 值（假设使用 DDPG 的 Critic 网络输出）
            if "critic" in self.model.__dict__:
                obs = self.locals["observations"]
                actions = self.locals["actions"]
                with torch.no_grad():
                    q_values = self.model.critic(obs, actions).numpy()
                self.q_values.append(np.mean(q_values))
            
            # 实时更新图表
            self._update_plots()
            
        return True
    
    def _update_plots(self):
        # 清空画布
        for ax in self.axs:
            ax.cla()
        
        # 绘制奖励曲线
        self.axs[0].plot(self.rewards, label="Episode Reward")
        self.axs[0].set_title("Training Reward")
        self.axs[0].set_xlabel("Episode")
        self.axs[0].set_ylabel("Reward")
        self.axs[0].legend()
        
        # 绘制 Q 值曲线
        if len(self.q_values) > 0:
            self.axs[1].plot(self.q_values, label="Average Q Value")
            self.axs[1].set_title("Critic Q Values")
            self.axs[1].set_xlabel("Step")
            self.axs[1].set_ylabel("Q Value")
            self.axs[1].legend()
        
        # 绘制动作分布
        if "actions" in self.locals:
            actions = np.array(self.locals["actions"])
            self.axs[2].hist(actions.flatten(), bins=20, alpha=0.6, label="Actions")
            self.axs[2].set_title("Action Distribution")
            self.axs[2].set_xlabel("Action Value")
            self.axs[2].set_ylabel("Frequency")
            self.axs[2].legend()
        
        plt.tight_layout()
        plt.pause(0.001)


# 训练模型
if __name__ == "__main__":
    env = MPCRLEnv()
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
    
    # 使用更深的网络结构
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],     # Actor 网络结构
            qf=[256, 256, 256]  # Critic 网络结构
        ),
        # 修正后的优化器参数（直接指定类及参数）
        actor_kwargs=dict(
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-4)
        ),
        critic_kwargs=dict(
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(lr=1e-3)
        )
    )
    
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        learning_rate=1e-3,  # 降低 Actor 学习率
        buffer_size=200000,  # 增大回放缓冲区
        batch_size=256,      # 增大批大小
        gamma=0.99,          # 增大折扣因子
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    monitor_callback = TrainingMonitorCallback(check_freq=100)
    model.learn(total_timesteps=1000000)
    model.save("rl_mpc_model")
    
    # 训练结束后保存图表
    plt.ioff()
    monitor_callback.fig.savefig("training_metrics.png")