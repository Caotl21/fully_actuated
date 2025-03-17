import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
from enum import Enum
from stable_baselines3 import DDPG

# ================== 轨迹类型定义 ==================
class TrajectoryType(Enum):
    STRAIGHT = 1     # 直线轨迹
    CIRCLE = 2       # 圆形轨迹
    DOUBLE_LANE = 3  # 双移线轨迹

# ================== 参数设置 ==================
# 车辆参数
L = 0.6          # 舵轮轴距 (m)
phi_max = np.deg2rad(45)  # 最大转向角
Ts = 0.05        # 缩短采样时间

# MPC参数
N = 15           # 延长预测时域
M = 8            # 增加控制时域
Q = np.diag([50, 5, 50, 5, 0.1])  # 调整后的状态权重
R = np.diag([0.01, 0.01])          # 降低输入权重
S = np.diag([0.1, 0.1])            # 输入增量权重

# 约束条件
w_max = np.array([3.0, 2.0])        # 放宽输入限制
delta_w_max = 1.0                   # 放宽输入增量
e_max = 0.3                        
I_max = 1.0                        

# 仿真参数
sim_time = 50.0                    
t = np.arange(0, sim_time, Ts)
selected_traj = TrajectoryType.CIRCLE

# ================== 轨迹生成器 ==================  
def generate_reference(traj_type, t):
    """
    生成参考轨迹
    返回: x_ref, y_ref, dx_ref, dy_ref, ddx_ref, ddy_ref
    """
    if traj_type == TrajectoryType.STRAIGHT:
        # 直线轨迹 (沿x轴匀速运动)
        v = 2.0  # m/s
        x = v * t
        #y = np.zeros_like(t)
        y = v * t
        dx = v * np.ones_like(t)
        dy = np.zeros_like(t)
        ddx = np.zeros_like(t)
        ddy = np.zeros_like(t)
    
    elif traj_type == TrajectoryType.CIRCLE:
        # 圆形轨迹 (半径3m，周期8s)
        R = 3.0
        omega = 2*np.pi/8
        x = R * np.cos(omega * t)
        y = R * np.sin(omega * t)
        dx = -R*omega*np.sin(omega*t)
        dy = R*omega*np.cos(omega*t)
        ddx = -R*omega**2*np.cos(omega*t)
        ddy = -R*omega**2*np.sin(omega*t)
    
    elif traj_type == TrajectoryType.DOUBLE_LANE:
        # 双移线轨迹 (两次换道)
        A = 1.0  # 换道幅度
        T = 10.0   # 换道周期
        x = np.linspace(0, 50, len(t))
        y = A * np.sin(2*np.pi*t/T) + A * np.sin(4*np.pi*t/T)
        
        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        ddx = np.gradient(dx, t)
        ddy = np.gradient(dy, t)
    
    else:
        raise ValueError("未知轨迹类型")
    
    # 曲率检查
    curvature = np.abs(ddx*dy - ddy*dx) / (dx**2 + dy**2)**1.5
    if np.any(1/curvature < L/np.tan(phi_max)):
        print("警告：轨迹包含不可达点！")

    return x, y, dx, dy, ddx, ddy


x_ref, y_ref, dx_ref, dy_ref, ddx_ref, ddy_ref = generate_reference(selected_traj, t)

# ================== 模型离散化 ==================
def discretize(A, B, dt):
    n = A.shape[0]
    M = np.block([[A, B], [np.zeros((B.shape[1], n+B.shape[1]))]])
    M_exp = expm(M*dt)
    Ad = M_exp[:n, :n]
    Bd = M_exp[:n, n:]
    return Ad, Bd

A_c = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [Ts, 0, Ts, 0, 1]
])
B_c = np.array([
    [0, 0],
    [-1, 0],
    [0, 0],
    [0, -1],
    [0, 0]
])

A_delta, B_delta = discretize(A_c, B_c, Ts)
D_delta = np.array([    [0, 0],
    [Ts, 0],
    [0, 0],
    [0, Ts],
    [0, 0]
])

# ================== 扰动观测器 ==================
class DisturbanceObserver:
    '''def __init__(self):
        self.d_hat = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.d_hat_history = []
    
    def update(self, error, u, dt=Ts):
        # 简单的一阶观测器
        alpha = 0.9  # 滤波系数
        self.d_hat = alpha * self.d_hat + (1-alpha) * (error - self.prev_error)/dt
        self.prev_error = error
        self.d_hat_history.append(self.d_hat.copy()) 
        return self.d_hat'''
    
    def __init__(self):
        self.d_hat = np.zeros(2)
        self.sigma = np.zeros(2)
        self.L = 5.0  # 滑模增益
        self.d_hat_history = []
    
    def update(self, e, de, dt = Ts):
        # 滑模面设计
        s = de + self.L * e
        # 等效控制法
        self.d_hat = np.clip(-self.L * e - self.sigma, -10, 10)
        self.sigma += self.L * self.d_hat * dt
        self.d_hat_history.append(self.d_hat.copy()) 
        return self.d_hat

    
    def get_history(self):
        """返回历史记录为 NumPy 数组"""
        return np.array(self.d_hat_history)

# ================== MPC控制器 ==================
class RobustMPC:
    def __init__(self):
        self.obs = DisturbanceObserver()
        self.u_prev = np.zeros(2)
        self.load_rl_model()
        self.current_Q_scale = 1.0  # 记录当前参数缩放因子
        self.current_R_scale = 1.0
        self.current_S_scale = 1.0  
        
        # 定义 Q, R, S 为可优化参数
        self.Q_param = cp.Parameter((5, 5), PSD=True)
        self.R_param = cp.Parameter((2, 2), PSD=True)
        self.S_param = cp.Parameter((2, 2), PSD=True)

        # 优化变量
        self.U = cp.Variable((M, 2))
        self.xi = cp.Parameter(5)
        self.xi_ref = cp.Parameter((N, 5))
        
    def build_problem(self):
        cost = 0
        constraints = []
        xi_pred = self.xi

        for i in range(N):
            if i < M:
                u = self.U[i]
            else:
                u = self.U[-1]

            # 状态预测
            xi_pred = A_delta @ xi_pred + B_delta @ u + D_delta @ self.xi_ref[i, 2:4]

            # 代价函数（使用参数化的 Q, R, S）
            cost += cp.quad_form(xi_pred - self.xi_ref[i], self.Q_param)
            if i < M:
                cost += cp.quad_form(u, self.R_param)
                if i > 0:
                    cost += cp.quad_form(u - self.U[i-1], self.S_param)

            # 约束条件
            constraints += [
                cp.abs(xi_pred[0]) <= e_max,
                cp.abs(xi_pred[2]) <= e_max,
                cp.abs(xi_pred[4]) <= I_max,
                cp.abs(u) <= w_max
            ]
            if i > 0 and i < M:
                constraints += [cp.abs(u - self.U[i-1]) <= delta_w_max]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        
        
    def load_rl_model(self):
        self.rl_model = DDPG.load("rl_mpc_model")  # 加载训练好的模型

    def update_parameters(self, state):
        # 使用 RL 模型预测动作（参数缩放因子）
        action, _ = self.rl_model.predict(state, deterministic=True)
        Q_scale, R_scale, S_scale = action
        self.current_Q_scale = Q_scale  # 更新参数值
        self.current_R_scale = R_scale
        self.current_S_scale = S_scale
        
        # 更新权重矩阵
        Q = np.diag([50*Q_scale, 5*Q_scale, 50*Q_scale, 5*Q_scale, 0.1*Q_scale])
        R = np.diag([0.01*R_scale, 0.01*R_scale])
        S = np.diag([0.1*S_scale, 0.1*S_scale])
        
        # 同步到 cvxpy 参数
        self.Q_param.value = Q
        self.R_param.value = R
        self.S_param.value = S
    
    def solve(self, xi_current, xi_ref, d_hat):
        self.xi.value = xi_current
        self.xi_ref.value = xi_ref
        
        try:
            self.problem.solve(solver=cp.OSQP, verbose=False)
            if self.problem.status == cp.OPTIMAL:
                u_opt = self.U.value[0] - d_hat  # 扰动补偿
                return np.clip(u_opt, -w_max, w_max)
            return np.zeros(2)
        except:
            return np.zeros(2)

# ================== 仿真初始化 ==================
mpc = RobustMPC()
xi = np.zeros(5)  # [e_x, Δe_x, e_y, Δe_y, I_e]
prev_error = np.zeros(2)

# 存储数据
x_hist, y_hist = [], []
u_hist, e_hist = [], []

# 实时绘图设置
plt.ion()
fig = plt.figure(figsize=(15, 10))

# ================== 主循环 ==================
for k in range(len(t)):
    # 参考轨迹生成
    xi_ref = np.zeros((N, 5))
    for i in range(N):
        if k+i < len(t):
            # 积分项计算
            if k+i > 0 and len(x_hist) >= k+i:
                I_e = np.sum(x_ref[:k+i] - x_hist[:k+i]) + np.sum(y_ref[:k+i] - y_hist[:k+i])
                I_e = np.clip(I_e, -I_max, I_max)
            else:
                I_e = 0
            
            xi_ref[i] = [0, 0, 0, 0, I_e]
            
    # 获取当前状态
    current_state = xi  # [e_x, Δe_x, e_y, Δe_y, I_e]

    # 使用 RL 模型更新 MPC 参数
    mpc.update_parameters(current_state)
    
    
    # 求解MPC
    current_error = np.array([xi[0], xi[2]])
    de = (current_error - prev_error) / Ts  # 误差的导数
    prev_error = current_error  # 更新 prev_error
    #d_hat = mpc.obs.update(current_error, mpc.u_prev)   #一阶观测器
    d_hat = mpc.obs.update(current_error, de)   #滑模观测器
    u_opt = mpc.solve(xi, xi_ref, d_hat)
    
    # 施加扰动 (5-6秒)
    #if 5.0 <= t[k] < 6.0:
    #    u_opt += np.array([0, 0.8])
    
    # 前馈补偿
    u_ff = np.array([ddx_ref[k], ddy_ref[k]])
    u_total = u_opt + 0.5*u_ff  # 前馈增益
    
    # 状态更新
    xi_next = A_delta @ xi + B_delta @ u_total + D_delta @ np.array([ddx_ref[k], ddy_ref[k]])
    
    # 积分抗饱和
    if np.linalg.norm([xi[0], xi[2]]) < 0.2:
        xi_next[4] = np.clip(xi[4] + Ts*(xi[0]+xi[2]), -I_max, I_max)
    else:
        xi_next[4] = 0
    
    # 记录数据
    x_hist.append(x_ref[k] + xi[0])
    y_hist.append(y_ref[k] + xi[2])
    u_hist.append(u_total)
    e_hist.append(xi[0]**2 + xi[2]**2)
    
    # 更新状态
    xi = xi_next
    mpc.u_prev = u_total
    
    # 实时绘图
    if k % 10 == 0:
        plt.clf()
        
        # 轨迹跟踪
        plt.subplot(2, 2, 1)
        plt.plot(x_ref, y_ref, 'r--', label='Reference')
        plt.plot(x_hist, y_hist, 'b-', label='Actual')
        plt.plot(xi_ref[:,0]+x_ref[k], xi_ref[:,2]+y_ref[k], 'g.', markersize=4, label='Predicted')
        plt.title(f'Tracking @ t={t[k]:.1f}s')
        plt.legend()
        
         # 在右上角显示参数
        param_text = f'Q_scale: {mpc.current_Q_scale:.2f}\nR_scale: {mpc.current_R_scale:.2f}\nS_scale: {mpc.current_S_scale:.2f}'
        plt.text(0.95, 0.95, param_text, transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # 误差分析
        plt.subplot(2, 2, 2)
        epsilon = 1e-6
        plt.plot(t[:k+1], np.array(e_hist)+epsilon, label='Squared Error')
        plt.yscale('log')
        plt.title('Tracking Error')
        plt.legend()
        
        # 控制输入
        plt.subplot(2, 2, 3)
        plt.plot(t[:k+1], np.array(u_hist)[:,0], label='w1')
        plt.plot(t[:k+1], np.array(u_hist)[:,1], label='w2')
        plt.ylim([-3.5, 3.5])
        plt.title('Control Inputs')
        plt.legend()
        
        # 扰动估计
        d_hat_history = mpc.obs.get_history()
        plt.subplot(2, 2, 4)
        plt.plot(t[:k+1], d_hat_history[:,0], label='d_x')
        plt.plot(t[:k+1], d_hat_history[:,1], label='d_y')
        plt.title('Disturbance Estimation')
        plt.legend()
        
        plt.tight_layout()
        plt.pause(0.001)

plt.ioff()
plt.show()